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

    def associate_iu_r2gen_kaggle_by_id(self):
        r2gen_splits_ids_reports = {
            split: [{sample["id"]: sample["report"]} for sample in samples]
            for split, samples in self.r2gen_ann.items()
        }

        kaggle_uids_mesh_impression = {
            line[0]: [{"MeSH": line[1], "report": line[6], "impression": line[7]}]
            for line in self.kaggle_iu_reports
        }

        unmatched = dict(train=[], val=[], test=[])
        matched = dict(train=[], val=[], test=[])
        for split, samples in r2gen_splits_ids_reports.items():
            for sample in samples:
                for r2gen_id, r2gen_report in sample.items():
                    uid = r2gen_id.split('_')[0].replace("CXR", "")
                    if uid in kaggle_uids_mesh_impression:
                        kaggle_report = kaggle_uids_mesh_impression[uid]["report"]
                        if r2gen_report == kaggle_report:
                            matched_info = {r2gen_id: {"MeSH": kaggle_uids_mesh_impression[uid]["MeSH"],
                                                       "impression": kaggle_uids_mesh_impression[uid]["impression"]}}
                            matched[split].append(matched_info)
                        else:
                            unmatched_info = {r2gen_id: {"r2gen_uid": uid, "r2gen_report": r2gen_report,
                                                         "kaggle_report": kaggle_report}}
                            unmatched[split].append(unmatched_info)
                    else:
                        unmatched_info = {
                            r2gen_id: {"r2gen_uid": uid, "r2gen_report": r2gen_report, "kaggle_report": ""}}
                        unmatched[split].append(unmatched_info)
