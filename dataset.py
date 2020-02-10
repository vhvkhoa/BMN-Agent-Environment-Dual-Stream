# -*- coding: utf-8 -*-
import os
import json

import numpy as np

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from utils import ioa_with_anchors, iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


def train_collate_fn(batch):
    batch_env_features, batch_agent_features, confidence_labels, start_labels, end_labels = zip(*batch)

    # Sort videos in batch by temporal lengths
    len_sorted_ids = sorted(range(len(batch_env_features)), key=lambda i: len(batch_env_features[i]))
    batch_env_features = [batch_env_features[i] for i in len_sorted_ids]
    batch_agent_features = [batch_agent_features[i] for i in len_sorted_ids]
    confidence_labels = [confidence_labels[i] for i in len_sorted_ids]
    start_labels = [start_labels[i] for i in len_sorted_ids]
    end_labels = [end_labels[i] for i in len_sorted_ids]

    # Create agent feature padding mask
    batch_agent_box_lengths = [[len(t_feature) for t_feature in agent_features] for agent_features in batch_agent_features]
    max_box_dim = torch.max(batch_agent_box_lengths)
    batch_agent_features_padding_mask = torch.arange(max_box_dim)[None, None, :] < batch_agent_box_lengths[:, :, None]

    # Declare important dimensions
    batch_size = batch_env_features
    max_temporal_dim = max([len(env_features) for env_features in batch_env_features])
    feature_dim = len(batch_env_features[0][0])
    
    # Pad environment features at temporal dimension
    padded_batch_env_features = torch.zeros(batch_size, max_temporal_dim, feature_dim)
    for i, env_features in enumerate(batch_env_features):
        for j, temporal_features in env_features:
            padded_batch_env_features[i][j] = temporal_features
    
    # Pad agent features at temporal and box dimension
    padded_batch_agent_features = torch.zeros(batch_size, max_temporal_dim, max_box_dim, feature_dim)
    for i, agent_features in enumerate(batch_agent_features):
        for j, temporal_features in enumerate(agent_features):
            for k, box_features in enumerate(temporal_features):
                padded_batch_agent_features[i][j][k] = box_features
    
    return padded_batch_env_features, padded_batch_agent_features, confidence_labels, start_labels, end_labels


def test_collate_fn(batch):
    return


class VideoDataSet(Dataset):
    def __init__(self, opt, split="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.split = split
        self.env_feature_dir = opt["env_feature_dir"]
        self.agent_feature_dir = opt["agent_feature_dir"]
        self.video_id_path = opt["video_id_path"]
        self.video_anno_path = opt["video_anno_path"]
        self._getDatasetDict()
        self._get_match_map()

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
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return env_features, agent_features, confidence_score, match_score_start, match_score_end
        else:
            return index, env_features, agent_features

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.temporal_scale + 1)]

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
        env_features = torch.transpose(torch.tensor([env_features_dict[t] for t in env_timestamps]), 0, 1).float()

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        agent_features_dict = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))
        agent_timestamps = sorted(agent_features_dict.keys(), key=lambda x: float(x))
        # agent_features = nn.utils.rnn.pad_sequence([agent_features_dict[t] for t in agent_timestamps], batch_first=True)
        agent_features = [agent_features_dict[t] for t in agent_timestamps]

        assert env_timestamps == agent_timestamps, 'Two streams must have same paces.'

        return env_timestamps, env_features, agent_features

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_names[index]
        video_info = self.event_dict[video_name]
        duration = video_info['duration']
        video_labels = video_info['events']  # the measurement is second, not frame

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
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
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
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, split="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True, collate_fn=train_collate_fn)
    for a, b, c, d in train_loader:
        print(a.shape, b.shape, c.shape, d.shape)
        break
