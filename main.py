import sys
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim

import opts
from models import EventDetection
from dataset import VideoDataSet, train_collate_fn, test_collate_fn
from loss_function import bmn_loss_func, get_mask
from post_processing import BMN_post_processing
from eval import evaluation_proposal

from config.defaults import get_cfg

sys.dont_write_bytecode = True


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (env_features, agent_features, agent_padding_masks, label_confidence, label_start, label_end) in enumerate(data_loader):
        env_features = env_features.cuda()
        agent_features = agent_features.cuda()
        agent_padding_masks = agent_padding_masks.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(env_features, agent_features, agent_padding_masks)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        print("Step %d:\n\tLoss: %f.\tPemreg Loss: %f.\tPemclr Loss: %f.\tTem Loss: %f." % (
            n_iter,
            loss[0].cpu().detach().numpy(),
            loss[2].cpu().detach().numpy(),
            loss[3].cpu().detach().numpy(),
            loss[1].cpu().detach().numpy()))

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))


def test_BMN(data_loader, model, epoch, bm_mask):
    model.eval()
    for indices, env_features, agent_features, agent_padding_masks in data_loader:
        env_features = env_features.cuda()
        agent_features = agent_features.cuda()
        agent_padding_masks = agent_padding_masks.cuda()

        confidence_map, start, end = model(env_features, agent_features, agent_padding_masks)

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")


def BMN_Train(cfg):
    model = EventDetection(cfg)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR,
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="train"),
                                               batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=8, pin_memory=True, collate_fn=train_collate_fn)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="validation"),
                                              batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                              num_workers=8, pin_memory=True, collate_fn=test_collate_fn)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(cfg.DATA.TEMPORAL_DIM)
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        scheduler.step()
        # test_BMN(test_loader, model, epoch, bm_mask)


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

            ####################################################################################################
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
            ########################################################################################################

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
        evaluation_proposal(cfg)


if __name__ == '__main__':
    cfg = get_cfg()
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(cfg)
