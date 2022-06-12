import json
import csv


class DataProcessor(object):
    def __init__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.r2gen_ann = json.loads(open(self.ann_path, 'r').read())
        # self.kaggle_iu_projections_path = args.kaggle_iu_projections_path
        # self.kaggle_iu_projections = json.loads(open(self.kaggle_iu_projections_path, 'r').read())
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.kaggle_iu_reports = csv.reader(self.kaggle_iu_reports_path)

    def associate_iu_r2gen_kaggle(self):
        r2gen_splits_ids = {r2gen_split: {v["id"]: v["report"]} for r2gen_splits, values in self.r2gen_ann.items() for
                            r2gen_split, v in values}
        print(self.kaggle_iu_reports[0])
