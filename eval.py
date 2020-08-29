# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
from evaluation import prop_eval
import pandas as pd
import requests
import pickle
import io
# from oct2py import octave


# octave.addpath('THUMOS14_evalkit_20150930')


'''
def standardize_results(video_dict, split='validation'):
    results = []
    for video_id, proposals in video_dict.items():
        for proposal in proposals:
            result_row = ' '.join([str(x) for x in [video_id] + proposal['segment'] + [proposal['score']]])
            results.append(result_row)
    return '\n'.join(results)
'''


'''
def run_evaluation(ground_truth_filename, proposal_filename,
                   tiou_thresholds=np.linspace(0.5, 1.0, 11),
                   subset='validation'):
    recalls = []
    for threshold in tiou_thresholds:
        eval_outputs = octave.TH14evalDet(proposal_filename, ground_truth_filename, subset, threshold)
        recalls.append(eval_outputs)
        print('Finished ', threshold)
    average_recall = np.mean(recalls)

    return average_recall
'''
def run_evaluation(proposal_filename, nr_proposals):
    frm_nums = pickle.load(open("evaluation/frm_num.pkl", 'rb'))
    rows = prop_eval.pkl2dataframe(frm_nums, 'evaluation/movie_fps.pkl', proposal_filename)
    daps_results = pd.DataFrame(rows, columns = ['f-end','f-init','score','video-frames','video-name'])

    # Retrieves and loads Thumos14 test set ground-truth.
    ground_truth_url = ('https://gist.githubusercontent.com/cabaf/'
                        'ed34a35ee4443b435c36de42c4547bd7/raw/'
                        '952f17b9cdc6aa4e6d696315ba75091224f5de97/'
                        'thumos14_test_groundtruth.csv')
    s = requests.get(ground_truth_url).content
    ground_truth = pd.read_csv(io.StringIO(s.decode('utf-8')),sep=' ')
    # Computes recall for different tiou thresholds at a fixed average number of proposals.
    recall, tiou_thresholds = prop_eval.recall_vs_tiou_thresholds(daps_results, ground_truth,
                                                        nr_proposals=nr_proposals,
                                                        tiou_thresholds=np.linspace(0.5, 1.0, 11))
    recall = np.mean(recall)
    return recall


def evaluate_proposals(cfg, nr_proposals_list=[50, 100, 200, 500, 1000]):
    '''
    with open(cfg.DATA.RESULT_PATH, 'r') as f:
        proposals_data = json.load(f)
    '''
    
    average_recalls = {}
    for nr_proposals in nr_proposals_list:
        '''
        nr_proposals_data = proposals_data['results'].copy()
        for video_name in nr_proposals_data.keys():
            proposals_list = sorted(nr_proposals_data[video_name], key=lambda x: x['score'], reverse=True)
            nr_proposals_data[video_name] = proposals_list[:nr_proposals]

        with open('evaluate_submission.txt', 'w') as f:
            f.write(standardize_results(nr_proposals_data))
        '''

        average_recalls[nr_proposals] = run_evaluation(cfg.DATA.RESULT_PATH, nr_proposals)

    for nr_proposals, average_recall in average_recalls.items():
        print("AR@%d is %f\t" % (nr_proposals, average_recall))

    return average_recalls[100]
