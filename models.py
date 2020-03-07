# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.defaults import get_cfg
from self_attention import ModifiedMultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, num_features, num_heads=4, dim_feedforward=1024, drop_out=0.1, activation='relu', num_layers=1, norm=None):
        super(TransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(num_features, num_heads, dim_feedforward, drop_out, activation)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, key_padding_mask=None):
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    key_padding_mask=key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = ModifiedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=key_padding_mask)[0]
        if key_padding_mask is not None:
            src2 = src2.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if key_padding_mask is not None:
            src = src.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        return src


class EventDetection(nn.Module):
    def __init__(self, cfg):
        super(EventDetection, self).__init__()

        self.agents_fuser = TransformerEncoder(cfg.DATA.FEATURE_DIM)
        self.agents_environment_fuser = TransformerEncoder(cfg.DATA.FEATURE_DIM)
        self.event_detector = BoundaryMatchingNetwork(cfg)

        self.attention_steps = cfg.TRAIN.ATTENTION_STEPS

    def forward(self, env_features, agent_features, lengths, env_masks, agent_masks):
        bsz, tmprl_sz, n_boxes, ft_sz = agent_features.size()
        step = self.attention_steps

        # Fuse all agents together at every temporal point
        len_idx, smpl_bgn, tmp_bsz = len(lengths) - 1, 0, bsz
        agent_fused_features = torch.zeros(tmp_bsz, tmprl_sz, ft_sz).cuda()

        if n_boxes > 0:
            while len_idx >= 0:
                smpl_end = min(lengths[len_idx], smpl_bgn + step)

                fuser_input = agent_features[:tmp_bsz, smpl_bgn:smpl_end].contiguous()
                fuser_input = fuser_input.view(-1, n_boxes, ft_sz).permute(1, 0, 2)
                attention_padding_masks = agent_masks[:tmp_bsz, smpl_bgn:smpl_end]
                attention_padding_masks = attention_padding_masks.contiguous().view(-1, n_boxes)

                keep_mask = (torch.sum(~attention_padding_masks, dim=-1) > 0)
                keep_indices = torch.masked_select(torch.arange(attention_padding_masks.size(0)).cuda(), keep_mask)

                if len(keep_indices) > 0:
                    fuser_input = fuser_input[:, keep_indices]
                    attention_padding_masks = attention_padding_masks[keep_indices]

                    padded_output = torch.zeros(tmp_bsz * (smpl_end - smpl_bgn), ft_sz).cuda()
                    fuser_output = self.agents_fuser(fuser_input, key_padding_mask=attention_padding_masks)
                    fuser_output = torch.sum(fuser_output, dim=0) / torch.sum(~attention_padding_masks, dim=-1, keepdim=True)
                    padded_output[keep_indices] = fuser_output
                    agent_fused_features[:tmp_bsz, smpl_bgn:smpl_end] = padded_output.view(tmp_bsz, -1, ft_sz)

                while len_idx >= 0 and smpl_end == lengths[len_idx]:
                    len_idx -= 1
                    tmp_bsz -= 1
                smpl_bgn = smpl_end

        env_agent_cat_features = torch.stack([env_features, agent_fused_features], dim=2)
        print(env_agent_cat_features.size())

        len_idx, smpl_bgn, tmp_bsz = len(lengths) - 1, 0, bsz
        context_features = torch.zeros(bsz, tmprl_sz, ft_sz).cuda()

        while len_idx >= 0:
            smpl_end = min(lengths[len_idx], smpl_bgn + step)

            fuser_input = env_agent_cat_features[:tmp_bsz, smpl_bgn:smpl_end].contiguous()
            fuser_input = fuser_input.view(-1, 2, ft_sz).permute(1, 0, 2)
            attention_padding_masks = env_masks[:tmp_bsz, smpl_bgn:smpl_end]
            attention_padding_masks = attention_padding_masks.contiguous().view(-1, 2)

            print(fuser_input.size())
            fuser_output = self.agents_environment_fuser(fuser_input, key_padding_mask=attention_padding_masks)
            fuser_output = torch.mean(fuser_output, dim=0)
            context_features[:tmp_bsz, smpl_bgn:smpl_end] = fuser_output.view(tmp_bsz, -1, ft_sz)

            while len_idx >= 0 and smpl_end == lengths[len_idx]:
                len_idx -= 1
                tmp_bsz -= 1
            smpl_bgn = smpl_end

        return self.event_detector(context_features.permute(0, 2, 1))


class BoundaryMatchingNetwork(nn.Module):
    def __init__(self, cfg):
        super(BoundaryMatchingNetwork, self).__init__()
        self.prop_boundary_ratio = cfg.BMN.PROP_BOUNDARY_RATIO
        self.num_sample = cfg.BMN.NUM_SAMPLES
        self.num_sample_perbin = cfg.BMN.NUM_SAMPLES_PER_BIN
        self.feat_dim = cfg.DATA.FEATURE_DIM
        self.temporal_dim = cfg.DATA.TEMPORAL_DIM

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)

        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)

        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(
            input_size[0],
            input_size[1],
            self.num_sample,
            self.temporal_dim,
            self.temporal_dim
        )
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.temporal_dim):
            mask_mat_vector = []
            for duration_index in range(self.temporal_dim):
                if start_index + duration_index < self.temporal_dim:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.temporal_dim, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.temporal_dim, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.temporal_dim, -1), requires_grad=False)


if __name__ == '__main__':
    cfg = get_cfg()
    model = EventDetection(cfg)

    batch_size = 1
    temporal_dim = 100
    box_dim = 4
    feature_dim = 2304

    env_input = torch.randn(batch_size, temporal_dim, feature_dim)
    agent_input = torch.randn(batch_size, temporal_dim, box_dim, feature_dim)
    agent_padding_mask = torch.tensor(np.random.randint(0, 1, (batch_size, temporal_dim, box_dim))).bool()

    a, b, c = model(env_input, agent_input, agent_padding_mask)
    print(a.shape, b.shape, c.shape)
