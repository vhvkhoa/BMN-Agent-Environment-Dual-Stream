import sys
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim

from models import EventDetection
from dataset import VideoDataSet, train_collate_fn, test_collate_fn
from loss_function import FocalLoss, bmn_loss_func, get_mask
from post_processing import BMN_post_processing
from utils import evaluate_proposals

from config.defaults import get_cfg

sys.dont_write_bytecode = True


def train_BMN(cfg, train_loader, test_loader, model, optimizer, epoch, focal_loss, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (env_features, agent_features, lengths, env_masks, agent_masks, label_confidence, label_start, label_end) in enumerate(train_loader):
        env_features = env_features.cuda()
        agent_features = agent_features.cuda()
        env_masks = env_masks.cuda()
        agent_masks = agent_masks.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(env_features, agent_features, lengths, env_masks, agent_masks)

        loss = bmn_loss_func(focal_loss, confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        print("Step %d:\tLoss: %f.\tTem Loss: %f.\tPem Cls Loss: %f." % (
            n_iter,
            loss[0].cpu().detach().numpy(),
            # loss[2].cpu().detach().numpy(),
            loss[1].cpu().detach().numpy(),
            loss[2].cpu().detach().numpy()))

        # epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[2].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        if n_iter % 1000 and n_iter != 0:
            evaluate(cfg, test_loader, model, epoch, n_iter)

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

    evaluate(cfg, test_loader, model, epoch, n_iter)


def evaluate(cfg, data_loader, model, epoch, n_iter=0):
    model.eval()
    with torch.no_grad():
        for video_name, env_features, agent_features, lengths, env_masks, agent_masks in data_loader:
            env_features = env_features.cuda()
            agent_features = agent_features.cuda()
            env_masks = env_masks.cuda()
            agent_masks = agent_masks.cuda()

            confidence_map, start, end = model(env_features, agent_features, lengths, env_masks, agent_masks)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

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
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)

    print("Post processing start")
    BMN_post_processing(cfg)
    print("Post processing finished")
    evaluate_proposals(cfg)

    with open(cfg.DATA.SCORE_PATH, 'r') as f:
        scores = json.load(f)

    if os.path.isfile(cfg.MODEL.BEST_RECORDS):
        with open(cfg.MODEL.BEST_RECORDS, 'r') as f:
            best_scores = json.load(f)

        for metric, sub_scores in scores.items():
            for i, score in enumerate(sub_scores):
                if score > best_scores[metric][i]:
                    state = {
                        'epoch': epoch + 1,
                        'iter': n_iter,
                        'state_dict': model.state_dict()
                    }
                    torch.save(state, os.path.join(cfg.MODEL.CHECKPOINT_DIR, "best_%s_%d.pth.tar" % (metric, i)))
    else:
        best_scores = scores

    with open(cfg.MODEL.BEST_RECORDS, 'w') as f:
        json.dump(best_scores, f)

    state = {
        'epoch': epoch + 1,
        'iter': n_iter,
        'state_dict': model.state_dict()
    }
    torch.save(state, os.path.join(cfg.MODEL.CHECKPOINT_DIR, "model_%d_%d.pth.tar" % (epoch + 1, n_iter)))


def BMN_Train(cfg):
    model = EventDetection(cfg)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR)
    focal_loss = FocalLoss()

    train_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="train"),
                                               batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=1, pin_memory=True, collate_fn=train_collate_fn)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True, collate_fn=test_collate_fn)

    bm_mask = get_mask(cfg.DATA.TEMPORAL_DIM)
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_BMN(cfg, train_loader, test_loader, model, optimizer, epoch, focal_loss, bm_mask)


def BMN_inference(cfg):
    model = EventDetection(cfg)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(os.path.join(cfg.MODEL.CHECKPOINT_DIR, "BMN_best.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False, collate_fn=train_collate_fn)
    tscale = cfg.DATA.TEMPORAL_DIM
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            max_start = max(start_scores)
            max_end = max(end_scores)

            #########################################################################
            # generate the set of start points and end points
            start_bins = np.zeros(len(start_scores))
            start_bins[0] = 1  # [1,0,0...,0,1] 首末两帧
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
            # 遍历起始分界点与结束分界点的组合
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
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(cfg):
    if cfg.MODE == "train":
        BMN_Train(cfg)
    elif cfg.MODE == "inference":
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference(cfg)
        print("Post processing start")
        BMN_post_processing(cfg)
        print("Post processing finished")
        evaluate_proposals(cfg)


if __name__ == '__main__':
    cfg = get_cfg()
    main(cfg)
