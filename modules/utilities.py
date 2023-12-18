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


def merge(r2gen_result_path, record_path, union_result_dir_name, union_result_file_name):
    with open(r2gen_result_path, 'r') as f:
        r2gen_result = json.load(f)
    ground_truth = [item['ground_truth'] for item in r2gen_result['test']]
    target_report = [item['report'] for item in r2gen_result['test']]
    
    with open(record_path, 'r') as f:
        record = json.load(f)
    refined_report = record['test']

    if not os.path.exists(union_result_dir_name):
        os.makedirs(union_result_dir_name)
    with open(union_result_dir_name + '/' + union_result_file_name, 'w') as f:
        for i, (gt, tg, rf) in enumerate(zip(ground_truth, target_report, refined_report)):
            f.write(str(i) + '\n')
            f.write('ground_truth: ' + gt + '\n')
            f.write('target_report: ' + tg + '\n')
            f.write('refined_report: ' + rf + '\n')
