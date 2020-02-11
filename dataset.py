# -*- coding: utf-8 -*-
import os
import json

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from utils import ioa_with_anchors, iou_with_anchors

from config.defaults import get_cfg


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


def train_collate_fn(batch):
    batch_env_features, batch_agent_features, confidence_labels, start_labels, end_labels = zip(*batch)
    for conf, sl, el in zip(confidence_labels, start_labels, end_labels):
        print(conf.size(), sl.size(), el.size())

    # Sort videos in batch by temporal lengths
    len_sorted_ids = sorted(range(len(batch_env_features)), key=lambda i: len(batch_env_features[i]))
    batch_env_features = [batch_env_features[i] for i in len_sorted_ids]
    batch_agent_features = [batch_agent_features[i] for i in len_sorted_ids]
    confidence_labels = torch.stack([confidence_labels[i] for i in len_sorted_ids])
    start_labels = torch.stack([start_labels[i] for i in len_sorted_ids])
    end_labels = torch.stack([end_labels[i] for i in len_sorted_ids])

    # Create agent feature padding mask
    batch_agent_box_lengths = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([len(t_feature) for t_feature in agent_features])
        for agent_features in batch_agent_features], batch_first=True
    )
    max_box_dim = torch.max(batch_agent_box_lengths).item()
    batch_agent_features_padding_mask = torch.arange(max_box_dim)[None, None, :] < batch_agent_box_lengths[:, :, None]

    # Pad environment features at temporal dimension
    padded_batch_env_features = pad_sequence(batch_env_features, batch_first=True)

    # Pad agent features at temporal and box dimension
    batch_size, max_temporal_dim, feature_dim = padded_batch_env_features.size()
    padded_batch_agent_features = torch.zeros(batch_size, max_temporal_dim, max_box_dim, feature_dim)

    for i, temporal_features in enumerate(batch_agent_features):
        for j, box_features in enumerate(temporal_features):
            if len(box_features) > 0:
                padded_batch_agent_features[i, j, :len(box_features)] = torch.tensor(box_features)

    return padded_batch_env_features, padded_batch_agent_features, batch_agent_features_padding_mask, confidence_labels, start_labels, end_labels


def test_collate_fn(batch):
    return


class VideoDataSet(Dataset):
    def __init__(self, cfg, split="train"):
        self.temporal_scale = cfg.DATA.TEMPORAL_SCALE  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.split = split
        self.env_feature_dir = cfg.DATA.ENV_FEATURE_DIR
        self.agent_feature_dir = cfg.DATA.AGENT_FEATURE_DIR
        self.video_id_path = cfg.DATA.VIDEO_ID_FILE
        self.video_anno_path = cfg.DATA.VIDEO_ANNOTATION_FILE

        self._getDatasetDict()

    def _getDatasetDict(self):
        self.video_names = load_json(self.video_id_path)
        annotations = load_json(self.video_anno_path)
        # Read event segments
        self.event_dict = {}
        for video_name in self.video_names:
            annotation = annotations[video_name]
            self.event_dict[video_name] = {'duration': annotation['duration'], 'events': annotation['timestamps']}

        print("Split: %s. Video numbers: %d" % (self.split, len(self.video_names)))

    def __getitem__(self, index):
        env_timestamps, env_features, agent_features = self._load_item(index)
        if self.split == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index)
            return env_features, agent_features, confidence_score, match_score_start, match_score_end
        else:
            return index, env_features, agent_features

    def _get_match_map(self, temporal_scale):
        match_map = []
        for idx in range(temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start

        anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(temporal_scale)]
        anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, temporal_scale + 1)]

        return match_map, anchor_xmin, anchor_xmax

    def _load_item(self, index):
        video_name = self.video_names[index]

        '''
        Read environment features at every timestamp
        Feature size: FxT
        T: number of timestamps
        F: feature size
        '''
        env_features_dict = load_json(os.path.join(self.env_feature_dir, video_name + '.json'))
        env_timestamps = sorted(env_features_dict.keys(), key=lambda x: float(x))
        env_features = torch.tensor([env_features_dict[t] for t in env_timestamps]).float().squeeze(1)

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        agent_features_dict = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))
        agent_timestamps = sorted(agent_features_dict.keys(), key=lambda x: float(x))
        agent_features = [agent_features_dict[t] for t in agent_timestamps]

        assert env_timestamps == agent_timestamps, 'Two streams must have same paces.'

        return env_timestamps, env_features, agent_features

    def _get_train_label(self, index):
        video_name = self.video_names[index]
        video_info = self.event_dict[video_name]
        duration = video_info['duration']
        video_labels = video_info['events']  # the measurement is second, not frame

        match_map, anchor_xmin, anchor_xmax = self._get_match_map(duration)

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info[0] / duration), 0)
            tmp_end = max(min(1, tmp_info[1] / duration), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                match_map[:, 0], match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.tensor(match_score_start)
        match_score_end = torch.tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_names)


if __name__ == '__main__':
    cfg = get_cfg()
    train_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="train"),
                                               batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=8, pin_memory=True, collate_fn=train_collate_fn)
    for a, b, c, d, e in train_loader:
        print(a.size(), b.size(), c.size(), d.size(), e.size())
        break
