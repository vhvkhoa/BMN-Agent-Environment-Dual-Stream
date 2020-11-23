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


class ProposalGenerator(object):
    def __init__(self, temporal_dim=None, max_duration=None, annotations=None):
        self.tscale = temporal_dim
        self.max_duration = max_duration
        self.annots = annotations  # For THUMOS only
        self.rescale_segment = self.rescale_segment_anet if self.annots is None else self.rescale_segment_thumos

    def rescale_segment_anet(self, start_index, end_index, video_name=None):
        return start_index / self.tscale, end_index / self.tscale

    def rescale_segment_thumos(self, start_index, end_index, video_name=None):
        b = self.annots[video_name]['start_snippet']
        d = self.annots[video_name]['master_snippet_duration']
        return (start_index + b) / d, (end_index + b) / d

    def __call__(self, start, end, iou, video_names):
        batch_props = []
        for i, video_name in enumerate(video_names):
            start_scores = start[i]
            end_scores = end[i]
            iou_scores = iou[i]

            # generate proposals
            new_props = []
            for idx in range(self.tscale):
                for jdx in range(idx, self.max_duration):
                    xmin, xmax = self.rescale_segment(idx, jdx + 1, video_name)
                    xmin_score = start_scores[idx]
                    xmax_score = end_scores[jdx]
                    iou_score = iou_scores[idx, jdx]
                    score = xmin_score * xmax_score * iou_score
                    new_props.append([xmin, xmax, xmin_score, xmax_score, iou_score, score])
            new_props = np.stack(new_props)
            batch_props.append(new_props)

        return batch_props
