# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def get_mask(tscale, duration):
    bm_mask = np.zeros((tscale, tscale))
    for idx in range(tscale):
        for jdx in range(idx, tscale):
            if jdx - idx < duration:
                bm_mask[idx, jdx] = 1
    bm_mask = np.expand_dims(np.expand_dims(bm_mask, 0), 1)
    return torch.Tensor(bm_mask)


def binary_logistic_loss(gt_scores, pred_anchors):
    """
    Calculate weighted binary logistic loss
    :param gt_scores: gt scores tensor
    :param pred_anchors: prediction score tensor
    :return: loss output tensor
    """
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)

    pmask = (gt_scores > 0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = pmask.size()[0]

    ratio = num_entries / max(num_positive, 1)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    neg_pred_anchors = 1.0 - pred_anchors + epsilon
    pred_anchors = pred_anchors + epsilon

    loss = coef_1 * pmask * torch.log(pred_anchors) + coef_0 * (1.0 - pmask) * torch.log(
        neg_pred_anchors)
    loss = -1.0 * torch.mean(loss)
    return loss


def IoU_loss(gt_iou, pred_iou, mask):
    """
    Calculate IoU loss
    :param gt_iou: gt IoU tensor
    :param pred_iou: prediction IoU tensor
    :return: loss output tensor
    """
    u_hmask = (gt_iou > 0.6).float()
    u_mmask = ((gt_iou <= 0.6) & (gt_iou > 0.2)).float()
    u_lmask = (gt_iou <= 0.2).float() * mask

    u_hmask = u_hmask.view(-1)
    u_mmask = u_mmask.view(-1)
    u_lmask = u_lmask.view(-1)

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = 1.0 * num_h / num_m
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())
    u_smmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_mmask
    u_smmask = (u_smmask > (1.0 - r_m)).float()

    r_l = 2.0 * num_h / num_l
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())

    u_slmask = torch.rand(u_hmask.size()[0], requires_grad=False).cuda() * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask

    gt_iou = gt_iou.view(-1)
    pred_iou = pred_iou.view(-1)

    iou_loss = F.smooth_l1_loss(pred_iou * iou_weights, gt_iou * iou_weights, reduction='none')
    iou_loss = torch.sum(iou_loss * iou_weights) / torch.max(torch.sum(iou_weights),
                                                             torch.Tensor([1.0]).cuda())
    return iou_loss


def dbg_loss_func(cfg, preds, gt_labels, bm_mask):
    action = preds['action']
    iou = preds['iou']
    prop_start = preds['prop_start']
    prop_end = preds['prop_end']

    gt_action, gt_start, gt_end, iou_label = gt_labels

    # calculate action loss
    loss_action = binary_logistic_loss(gt_action, action)

    # calculate IoU loss
    iou_losses = 0.0
    batch_size = iou_label.shape[0]
    tmp_mask = bm_mask.repeat(batch_size, 1, 1, 1) > 0
    for i in range(batch_size):
        iou_loss = IoU_loss(iou_label[i:i + 1], iou[i:i + 1], bm_mask)
        iou_losses += iou_loss
    loss_iou = iou_losses / batch_size * 10.0

    # calculate starting and ending map loss
    gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, cfg.DATA.TEMPORAL_DIM)
    gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, cfg.DATA.TEMPORAL_DIM, 1)
    loss_start = binary_logistic_loss(
        torch.masked_select(gt_start, tmp_mask),
        torch.masked_select(prop_start, tmp_mask)
    )
    loss_end = binary_logistic_loss(
        torch.masked_select(gt_end, tmp_mask),
        torch.masked_select(prop_end, tmp_mask)
    )

    # total loss
    loss = 2.0 * loss_action + loss_iou + loss_start + loss_end
    return loss, loss_action, loss_iou, loss_start, loss_end


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):

    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask

    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):

    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss
