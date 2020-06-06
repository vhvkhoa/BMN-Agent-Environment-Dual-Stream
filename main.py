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


def train_BMN(cfg, train_loader, test_loader, model, optimizer, epoch, bm_mask, writer, checkpoint_dir):
    model.train()
    optimizer.zero_grad()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    period_loss = [0] * 4
    last_period_size = len(train_loader) % cfg.TRAIN.STEP_PERIOD
    last_period_start = cfg.TRAIN.STEP_PERIOD * (len(train_loader) // cfg.TRAIN.STEP_PERIOD)

    for n_iter, (env_features, agent_features, agent_masks, label_confidence, label_start, label_end) in enumerate(tqdm(train_loader)):
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

        confidence_map, start, end = model(env_features, agent_features, agent_masks)

        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        period_size = cfg.TRAIN.STEP_PERIOD if n_iter < last_period_start else last_period_size
        total_loss = loss[0] / period_size
        total_loss.backward()

        loss = [l.cpu().detach().numpy() / cfg.TRAIN.STEP_PERIOD for l in loss]
        period_loss = [l + pl for l, pl in zip(loss, period_loss)]

        if (n_iter + 1) % cfg.TRAIN.STEP_PERIOD != 0 and n_iter != (len(train_loader) - 1):
            continue

        optimizer.step()
        optimizer.zero_grad()

        '''
        print("Step %d:\tLoss: %f.\tTem Loss: %f.\tPem RegLoss: %f.\tPem ClsLoss: %f" % (
            n_iter,
            loss[0],
            loss[1],
            loss[2],
            loss[3]
        ))
        '''

        epoch_loss += period_loss[0]
        epoch_tem_loss += period_loss[1]
        epoch_pemclr_loss += period_loss[2]
        epoch_pemreg_loss += period_loss[3]

        '''
        writer.add_scalar('Loss', loss[0], epoch * len(train_loader) + n_iter)
        writer.add_scalar('TemLoss', loss[1], epoch * len(train_loader) + n_iter)
        writer.add_scalar('PemLoss Regression', loss[2], epoch * len(train_loader) + n_iter)
        writer.add_scalar('PemLoss Classification', loss[3], epoch * len(train_loader) + n_iter)
        '''
        write_step = epoch * len(train_loader) + n_iter
        writer.add_scalar('Loss', period_loss[0], write_step)
        writer.add_scalar('TemLoss', period_loss[1], write_step)
        writer.add_scalar('PemLoss Regression', period_loss[2], write_step)
        writer.add_scalar('PemLoss Classification', period_loss[3], write_step)
        period_loss = [0] * 4

        # if n_iter % 1000 == 0:  # and n_iter != 0:
        #     evaluate(cfg, test_loader, model, epoch, n_iter)

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

    evaluate(cfg, test_loader, model, epoch, writer, checkpoint_dir)


def evaluate(cfg, data_loader, model, epoch, writer, checkpoint_dir):
    model.eval()
    with torch.no_grad():
        for video_name, env_features, agent_features, agent_masks in tqdm(data_loader):
            video_name = video_name[0]
            env_features = env_features.cuda()
            agent_features = agent_features.cuda()
            agent_masks = agent_masks.cuda()

            confidence_map, start, end = model(
                env_features,
                agent_features,
                agent_masks
            )

            confidence_map = confidence_map.detach().cpu().numpy()
            start = start.detach().cpu().numpy()
            end = end.detach().cpu().numpy()

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

    print("Post processing start")
    BMN_post_processing(cfg)
    print("Post processing finished")
    auc_score = evaluate_proposals(cfg)
    writer.add_scalar('AUC', auc_score, epoch)

    if epoch == 0:
        scores = []
    else:
        with open(cfg.MODEL.SCORE_PATH, 'r') as f:
            scores = json.load(f)

    if len(scores) == 0 or auc_score > max(scores):
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(checkpoint_dir, "best_%s.pth" % 'auc'))

    scores.append(auc_score)

    with open(cfg.MODEL.SCORE_PATH, 'w') as f:
        json.dump(scores, f)

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict()
    }
    torch.save(state, os.path.join(checkpoint_dir, "model_%d.pth" % (epoch + 1)))


def BMN_Train(cfg):
    model = EventDetection(cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPU_IDS).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR)

    exp_id = max([0] + [int(run.split('_')[-1]) for run in os.listdir(cfg.TRAIN.LOG_DIR)]) + 1
    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, 'run_' + str(exp_id))
    writer = SummaryWriter(log_dir)

    checkpoint_dir = os.path.join(cfg.MODEL.CHECKPOINT_DIR, 'checkpoint_' + str(exp_id))
    assert not os.path.isdir(checkpoint_dir), 'Checkpoint directory %s has already been created.' % checkpoint_dir
    os.makedirs(checkpoint_dir)

    train_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="training"),
                                               batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=1, pin_memory=True, collate_fn=train_collate_fn)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True, drop_last=False, collate_fn=test_collate_fn)

    bm_mask = get_mask(cfg.DATA.TEMPORAL_DIM)
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_BMN(cfg, train_loader, test_loader, model, optimizer, epoch, bm_mask, writer, checkpoint_dir)


def BMN_inference(cfg):
    model = EventDetection(cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPU_IDS).cuda()
    checkpoint = torch.load(cfg.TEST.CHECKPOINT_PATH)
    print('Loaded model at epoch %d.' % checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split=cfg.MODE),
                                              batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True, drop_last=False, collate_fn=test_collate_fn)
    tscale = cfg.DATA.TEMPORAL_DIM
    with torch.no_grad():
        for video_name, env_features, agent_features, agent_masks in tqdm(test_loader):
            video_name = video_name[0]
            env_features = env_features.cuda()
            agent_features = agent_features.cuda()
            agent_masks = agent_masks.cuda()

            confidence_map, start, end = model(
                env_features,
                agent_features,
                agent_masks
            )

            confidence_map = confidence_map.detach().cpu().numpy()
            start = start.detach().cpu().numpy()
            end = end.detach().cpu().numpy()

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

    print("Post processing start")
    BMN_post_processing(cfg)
    print("Post processing finished")


def main(cfg):
    if cfg.MODE in ["train", "training"]:
        BMN_Train(cfg)
    else:
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
