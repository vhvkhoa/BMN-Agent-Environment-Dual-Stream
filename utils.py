import numpy as np
import subprocess
import os


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def generate_proposals(batch_size, start, end, confidence_map):
    batch_props = []
    for i in range(batch_size):
        start_scores = start[i]
        end_scores = end[i]
        clr_confidence = (confidence_map[i][1])
        reg_confidence = (confidence_map[i][0])

        max_start = max(start_scores)
        max_end = max(end_scores)

        tscale = len(start_scores)

        # generate the set of start points and end points
        start_bins = np.zeros(tscale)
        start_bins[0] = 1  # [1,0,0...,0,1]
        for idx in range(1, tscale - 1):
            if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                start_bins[idx] = 1
            elif start_scores[idx] > (0.5 * max_start):
                start_bins[idx] = 1

        end_bins = np.zeros(len(end_scores))
        end_bins[-1] = 1
        for idx in range(1, tscale - 1):
            if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                end_bins[idx] = 1
            elif end_scores[idx] > (0.5 * max_end):
                end_bins[idx] = 1

        # generate proposals
        new_props = []
        for idx in range(tscale):
            for jdx in range(tscale):
                start_index = jdx
                end_index = start_index + idx + 1
                if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                    xmin = start_index / tscale
                    xmax = end_index / tscale
                    xmin_score = start_scores[start_index]
                    xmax_score = end_scores[end_index]
                    clr_score = clr_confidence[idx, jdx]
                    reg_score = reg_confidence[idx, jdx]
                    score = xmin_score * xmax_score * clr_score * reg_score
                    new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
        new_props = np.stack(new_props)
        batch_props.append(new_props)

    return batch_props
