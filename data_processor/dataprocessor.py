import json


class DataProcessor:
    def __int__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.r2gen_ann = json.loads(open(self.ann_path, 'r').read())
        self.kaggle_iu_projections_path = args.kaggle_iu_projections_path
        self.kaggle_iu_projections = json.loads(open(self.kaggle_iu_projections_path, 'r').read())
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.kaggle_iu_reports = json.loads(open(self.kaggle_iu_reports_path, 'r').read())

    def associateIuxrayR2genToKaggle(self):
        r2gen_splits_ids = {r2gen_split: [v["id"]] for r2gen_splits, values in self.r2gen_ann.items() for r2gen_split, v
                            in values}
