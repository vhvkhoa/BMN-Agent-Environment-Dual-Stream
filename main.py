import sys
import os
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import EventDetection
from dataset import VideoDataSet, train_collate_fn, test_collate_fn
from loss_function import bmn_loss_func, get_mask
from post_processing import BMN_post_processing
# from utils import evaluate_proposals
from eval import evaluate_proposals

from config.defaults import get_cfg

sys.dont_write_bytecode = True


class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = EventDetection(cfg)
        self.model = torch.nn.DataParallel(self.model, device_ids=cfg.GPU_IDS).cuda()
        if cfg.MODE not in ['train', 'training']:  # TODO: add condition for resume feature.
            checkpoint = torch.load(cfg.TEST.CHECKPOINT_PATH)
            print('Loaded model at epoch %d.' % checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['state_dict'])

        if cfg.MODE in ['train', 'training']:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.TRAIN.LR)

    def inference(self, data_loader=None, split=None):
        if data_loader is None:
            data_loader = torch.utils.data.DataLoader(
                VideoDataSet(cfg, split=split),
                batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False, collate_fn=test_collate_fn)

        tscale = cfg.DATA.TEMPORAL_DIM
        self.model.eval()
        with torch.no_grad():
            for video_names, env_features, agent_features, agent_masks in tqdm(data_loader):
                video_name = video_names[0]

                env_features = env_features.cuda() if cfg.USE_ENV else None
                agent_features = agent_features.cuda() if cfg.USE_AGENT else None
                agent_masks = agent_masks.cuda() if cfg.USE_AGENT else None

                confidence_map, start, end = self.model(env_features, agent_features, agent_masks)

                confidence_map = confidence_map.cpu().numpy()
                start = start.cpu().numpy()
                end = end.cpu().numpy()

                # print(start.shape,end.shape,confidence_map.shape)
                start_scores = start[0]
                end_scores = end[0]
                clr_confidence = (confidence_map[0][1])
                reg_confidence = (confidence_map[0][0])

                max_start = max(start_scores)
                max_end = max(end_scores)

                tscale = len(start_scores)

                #########################################################################
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
                #########################################################################

                #########################################################################
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
                #########################################################################

                col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv("./outputs/BMN_results/" + video_name + ".csv", index=False)

    def epoch_train(self, data_loader, bm_mask, epoch, writer):
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad()
        loss_names = ['Loss', 'TemLoss', 'PemLoss Regression', 'PemLoss Classification']
        epoch_losses = [0] * 4
        period_losses = [0] * 4
        last_period_size = len(data_loader) % cfg.TRAIN.STEP_PERIOD
        last_period_start = cfg.TRAIN.STEP_PERIOD * (len(data_loader) // cfg.TRAIN.STEP_PERIOD)

        for n_iter, (env_features, agent_features, agent_masks, label_confidence, label_start, label_end) in enumerate(tqdm(data_loader)):
            if cfg.USE_ENV:
                env_features = env_features.cuda()
            else:
                env_features = None
            if cfg.USE_AGENT:
                agent_features = agent_features.cuda()
                agent_masks = agent_masks.cuda()
            else:
                agent_features = None
                agent_masks = None
            label_start = label_start.cuda()
            label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()

            confidence_map, start, end = self.model(env_features, agent_features, agent_masks)

            losses = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)
            period_size = cfg.TRAIN.STEP_PERIOD if n_iter < last_period_start else last_period_size
            total_loss = losses[0] / period_size
            total_loss.backward()

            losses = [loss.cpu().detach().numpy() / cfg.TRAIN.STEP_PERIOD for loss in losses]
            period_losses = [l + pl for l, pl in zip(losses, period_losses)]

            if (n_iter + 1) % cfg.TRAIN.STEP_PERIOD != 0 and n_iter != (len(data_loader) - 1):
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_losses = [el + pl for el, pl in zip(epoch_losses, period_losses)]

            write_step = epoch * len(data_loader) + n_iter
            for i, loss_name in enumerate(loss_names):
                writer.add_scalar(loss_name, period_losses[i], write_step)
            period_losses = [0] * 4

        print(
            "BMN training loss(epoch %d): tem_loss: %.03f, pem reg_loss: %.03f, pem cls_loss: %.03f, total_loss: %.03f" % (
                epoch, epoch_losses[1] / (n_iter + 1),
                epoch_losses[2] / (n_iter + 1),
                epoch_losses[3] / (n_iter + 1),
                epoch_losses[0] / (n_iter + 1)))

    def evaluate(self, data_loader=None, split=None):
        self.inference(data_loader, split)

        print("Post processing start")
        BMN_post_processing(self.cfg)
        print("Post processing finished")
        auc_score = evaluate_proposals(self.cfg)
        return auc_score

    def train(self, n_epochs):
        exp_id = max([0] + [int(run.split('_')[-1]) for run in os.listdir(self.cfg.TRAIN.LOG_DIR)]) + 1
        log_dir = os.path.join(self.cfg.TRAIN.LOG_DIR, 'run_' + str(exp_id))
        writer = SummaryWriter(log_dir)
        checkpoint_dir = os.path.join(self.cfg.MODEL.CHECKPOINT_DIR, 'checkpoint_' + str(exp_id))
        assert not os.path.isdir(checkpoint_dir), 'Checkpoint directory %s has already been created.' % checkpoint_dir
        os.makedirs(checkpoint_dir)

        train_loader = torch.utils.data.DataLoader(
            VideoDataSet(cfg, split="training"),
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
            num_workers=12, pin_memory=True, collate_fn=train_collate_fn)

        eval_loader = torch.utils.data.DataLoader(
            VideoDataSet(cfg, split="validation"),
            batch_size=1, shuffle=False,
            num_workers=12, pin_memory=True, drop_last=False, collate_fn=test_collate_fn)

        bm_mask = get_mask(cfg.DATA.TEMPORAL_DIM).cuda()
        scores = []
        for epoch in range(n_epochs):
            self.epoch_train(train_loader, bm_mask, epoch, writer)
            auc_score = self.evaluate(eval_loader)

            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict()
            }
            if len(scores) == 0 or auc_score > max(scores):
                torch.save(state, os.path.join(checkpoint_dir, "best_%s.pth" % 'auc'))
            torch.save(state, os.path.join(checkpoint_dir, "model_%d.pth" % (epoch + 1)))

            writer.add_scalar('AUC', auc_score, epoch)
            scores.append(auc_score)


def main(cfg):
    solver = Solver(cfg)
    if cfg.MODE in ["train", "training"]:
        solver.train(cfg.TRAIN.NUM_EPOCHS)
    elif cfg.MODE in ['validate', 'validation']:
        solver.evaluate(split='validation')
    elif cfg.MODE in ['test', 'testing']:
        solver.inference(split='testing')


if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)
