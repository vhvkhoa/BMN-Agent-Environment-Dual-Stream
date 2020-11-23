# -*- coding: utf-8 -*-
import os
import json

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from utils import ioa_with_anchors, iou_with_anchors

from config.defaults import get_cfg


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class Collator(object):
    def __init__(self, cfg, mode):
        self.is_train = mode in ['train', 'training']
        self.feat_names = ['env_feats', 'agent_feats', 'box_lens']
        if self.is_train:
            self.label_names = ['action_scores', 'start_scores', 'end_scores', 'iou_scores']
            self.batch_names = self.feat_names + self.label_names
        else:
            self.label_names = []
            self.batch_names = ['video_ids'] + self.feat_names
        self.tmp_dim = cfg.DATA.TEMPORAL_DIM
        self.feat_dim = cfg.DATA.FEATURE_DIM

    def process_features(self, bsz, env_feats, agent_feats, box_lens):
        if env_feats[0] is not None:
            env_feats = torch.stack(env_feats)
        else:
            env_feats = None

        # Make new order to inputs by their lengths (long-to-short)
        if agent_feats[0] is not None:
            box_lens = torch.stack(box_lens, dim=0)

            max_box_dim = torch.max(box_lens).item()
            # Make padding mask for self-attention
            agent_mask = torch.arange(max_box_dim)[None, None, :] >= box_lens[:, :, None]

            # Pad agent features at temporal and box dimension
            pad_agent_feats = torch.zeros(bsz, self.tmp_dim, max_box_dim, self.feat_dim)
            for i, temporal_features in enumerate(agent_feats):
                for j, box_features in enumerate(temporal_features):
                    if len(box_features) > 0:
                        pad_agent_feats[i, j, :len(box_features)] = torch.tensor(box_features)
        else:
            pad_agent_feats = None
            agent_mask = None
        return env_feats, pad_agent_feats, agent_mask

    def __call__(self, batch):
        input_batch = dict(zip(self.batch_names, zip(*batch)))
        bsz = len(input_batch['env_feats'])
        output_batch = [] if self.is_train else [input_batch['video_ids']]

        # Process environment and agent features
        input_feats = [input_batch[feat_name] for feat_name in self.feat_names]
        output_batch.extend(self.process_features(bsz, *input_feats))

        if len(self.label_names) > 0:
            gt_labels = []
            for label_name in self.label_names:
                gt_labels.append(torch.stack(input_batch[label_name]))
            output_batch.append(gt_labels)
        return output_batch


class VideoDataSet(Dataset):
    def __init__(self, cfg, split='training'):
        self.split = split
        # self.video_anno_path = cfg.DATA.ANNOTATION_FILE
        self.temporal_dim = cfg.DATA.TEMPORAL_DIM
        self.max_duration = cfg.DATA.MAX_DURATION
        self.temporal_gap = 1. / self.temporal_dim
        self.env_feature_dir = cfg.DATA.ENV_FEATURE_DIR
        self.agent_feature_dir = cfg.DATA.AGENT_FEATURE_DIR

        self.use_env = cfg.USE_ENV
        self.use_agent = cfg.USE_AGENT

        if split in ['train', 'training']:
            self.video_anno_path = cfg.TRAIN.ANNOTATION_FILE
            self._get_match_map()

        elif self.split in ['validation']:
            self.video_anno_path = cfg.VAL.ANNOTATION_FILE
        elif self.split in ['test', 'testing']:
            self.video_anno_path = cfg.TEST.ANNOTATION_FILE
        self.video_prefix = 'v_' if cfg.DATASET == 'anet' else ''

        self._get_dataset()

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_dim):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.max_duration + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map

        # self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_dim)]
        # self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.temporal_dim + 1)]
        self.anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_dim)]
        self.anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_dim + 1)]

    def _get_dataset(self):
        annotations = load_json(self.video_anno_path)['database']

        # Read event segments
        self.event_dict = {}
        self.video_ids = []

        for video_id, annotation in annotations.items():
            if annotation['subset'] != self.split:
                continue
            self.event_dict[video_id] = {
                'duration': annotation['duration'],
                'events': annotation['annotations']
                # 'events': annotation['timestamps']
            }
            self.video_ids.append(video_id)

        print("Split: %s. Dataset size: %d" % (self.split, len(self.video_ids)))

    def __getitem__(self, index):
        env_features, agent_features, box_lengths = self._load_item(index)
        if self.split == 'training':
            action_scores, start_scores, end_scores, iou_scores = self._get_train_label(index)
            return env_features, agent_features, box_lengths, action_scores, start_scores, end_scores, iou_scores
        else:
            return self.video_ids[index], env_features, agent_features, box_lengths

    def _load_item(self, index):
        video_name = self.video_prefix + self.video_ids[index]

        '''
        Read environment features at every timestamp
        Feature size: TxF
        T: number of timestamps
        F: feature size
        '''
        if self.use_env is True:
            env_features = load_json(os.path.join(self.env_feature_dir, video_name + '.json'))['video_features']
            # env_segments = [env['segment'] for env in env_features]
            env_features = torch.tensor([feature['features'] for feature in env_features]).float().squeeze(1)
        else:
            env_features = None

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        if self.use_agent is True:
            agent_features = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))['video_features']
            # agent_segments = [feature['segment'] for feature in agent_features]
            agent_features = [feature['features'] for feature in agent_features]
            # Create and pad agent_box_lengths if train
            box_lengths = torch.tensor([len(x) for x in agent_features])
        else:
            agent_features = None
            box_lengths = None

        # assert env_segments == agent_segments and len(env_segments) == 100, 'Two streams must have 100 segments.'

        return env_features, agent_features, box_lengths

    def _get_train_label(self, index):
        video_id = self.video_ids[index]
        video_info = self.event_dict[video_id]
        video_labels = video_info['events']  # the measurement is second, not frame
        duration = video_info['duration']

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / duration), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / duration), 0)
            gt_bbox.append([tmp_start, tmp_end])
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.temporal_gap, 0.1 * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_action = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_action.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_xmins, gt_xmaxs)))
        match_score_action = torch.Tensor(np.array(match_score_action)).unsqueeze(0)

        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_start = torch.Tensor(np.array(match_score_start)).unsqueeze(0)

        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_end = torch.Tensor(np.array(match_score_end)).unsqueeze(0)

        # gen iou_labels
        iou_labels = np.zeros([self.temporal_dim, self.temporal_dim])
        for i in range(self.temporal_dim):
            for j in range(i, self.temporal_dim):
                iou_labels[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        iou_labels = torch.Tensor(iou_labels).unsqueeze(0)

        return match_score_action, match_score_start, match_score_end, iou_labels

    def __len__(self):
        return len(self.video_ids)
