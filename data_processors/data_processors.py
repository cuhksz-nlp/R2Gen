import csv
import json


class DataProcessor(object):
    def __init__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.r2gen_ann = json.loads(open(self.r2gen_ann_path, 'r').read())
        self.kaggle_iu_projections_path = args.kaggle_iu_projections_path
        # self.kaggle_iu_projections = json.loads(open(self.kaggle_iu_projections_path, 'r').read())
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.kaggle_iu_reports = csv.reader(open(self.kaggle_iu_reports_path, 'r'))
        self.kaggle_iu_reports_header = next(self.kaggle_iu_reports)

    def associate_iu_r2gen_kaggle(self):
        r2gen_splits_ids_reports = {
            r2gen_split: [{sample["id"]: sample["report"]} for sample in samples]
            for r2gen_split, samples in self.r2gen_ann.items()
        }

        kaggle_uids_mesh_impression = {
            line[0]: [{"MeSH": line[1], "report": line[6], "impression": line[7]}]
            for line in self.kaggle_iu_reports
        }


