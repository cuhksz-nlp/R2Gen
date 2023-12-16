import os
import json
from modules.metrics import compute_scores


def evaluate(result, dataset_type):
    gts = {'train': [], 'val': [], 'test': []}
    res = {'train': [], 'val': [], 'test': []}
    for t in dataset_type:
        gts[t] = [item['ground_truth'] for item in result[t]]
        res[t] = [item['report'] for item in result[t]]
    
    score = {}
    for t in dataset_type:
        met = compute_scores({i: [gt] for i, gt in enumerate(gts[t])}, {i: [re] for i, re in enumerate(res[t])})
        score.update(**{t + '_' + k: v for k, v in met.items()})
    return score


def store(data, dir_name, file_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(dir_name + '/' + file_name, 'w') as f:
        json.dump(data, f)
