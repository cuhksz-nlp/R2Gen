import csv
import json
import os

from analytics.analyze import Analyze


class DataProcessor(object):
    # exp setup
    def __init__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.iu_mesh_impression_path_split = args.iu_mesh_impression_path.replace(".json", "_split.json")
        self.create_r2gen_kaggle_association = args.create_r2gen_kaggle_association
        self.iu_mesh_impression_split = dict()
        if self.create_r2gen_kaggle_association == 0 and os.path.exists(self.iu_mesh_impression_path_split):
            self.iu_mesh_impression_split = json.loads(open(self.iu_mesh_impression_path_split, 'r').read())
        else:
            self.iu_mesh_impression_split = self.associate_iu_r2gen_kaggle_by_id()
        self.analyze = Analyze(args)

    def associate_iu_r2gen_kaggle_by_id(self):
        kaggle_iu_reports = csv.reader(open(self.kaggle_iu_reports_path, 'r'))
        r2gen_ann = json.loads(open(self.r2gen_ann_path, 'r').read())
        next(kaggle_iu_reports)
        r2gen_splits_ids_reports = {
            split: [{sample["id"]: sample["report"]} for sample in samples]
            for split, samples in r2gen_ann.items()
        }

        kaggle_uids_mesh_impression = {
            line[0]: {"MeSH": line[1], "report": line[6], "impression": line[7]}
            for line in kaggle_iu_reports
        }

        unmatched_split = dict(train=[], val=[], test=[])
        matched_split = dict(train={}, val={}, test={})
        for split, samples in r2gen_splits_ids_reports.items():
            for sample in samples:
                for r2gen_id, r2gen_report in sample.items():
                    uid = r2gen_id.split('_')[0].replace("CXR", "")
                    if uid in kaggle_uids_mesh_impression:
                        kaggle_report = kaggle_uids_mesh_impression[uid]["report"]
                        if r2gen_report == kaggle_report:
                            iu_mesh = kaggle_uids_mesh_impression[uid]["MeSH"]
                            mesh_text = ""
                            attr_text = ""
                            mesh_attr_text = ""
                            for mesh_info in iu_mesh.split(';'):
                                if 'normal' != mesh_info and 'No Indexing' != mesh_info:
                                    mesh_attr = mesh_info.split('/')
                                    seq_mesh_text = ""
                                    seq_attr_text = ""
                                    if ',' in mesh_attr[0]:
                                        for ma in mesh_attr[0].split(','):
                                            seq_mesh_text += " <mesh:{}>".format(ma.strip().replace(' ', '_'))
                                    else:
                                        seq_mesh_text = " <mesh:{}>".format(mesh_attr[0].strip().replace(' ', '_'))
                                    mesh_text += seq_mesh_text

                                    if len(mesh_attr) == 2:
                                        seq_attr_text = " <attr:{}>".format(mesh_attr[1].strip().replace(' ', '_'))
                                    elif len(mesh_attr) > 2:
                                        for i in range(1, len(mesh_attr)):
                                            seq_attr_text += " <attr:{}>".format(mesh_attr[i].strip().replace(' ', '_'))
                                    attr_text += seq_attr_text

                                    mesh_attr_text += "{}{}".format(seq_mesh_text, seq_attr_text)
                            matched_split[split][r2gen_id] = {
                                "iu_mesh": iu_mesh, "mesh": mesh_text, "attr": attr_text,
                                "mesh_attr": mesh_attr_text,
                                "impression": kaggle_uids_mesh_impression[uid]["impression"]}
                            # matched[split].append(matched_info)
                        else:
                            unmatched_split[split][r2gen_id] = {"r2gen_uid": uid, "r2gen_report": r2gen_report,
                                                          "kaggle_report": kaggle_report}
                            # unmatched[split].append(unmatched_info)
                    else:
                        unmatched_info = {
                            r2gen_id: {"r2gen_uid": uid, "r2gen_report": r2gen_report, "kaggle_report": ""}}
                        unmatched_split[split].append(unmatched_info)
        if not os.path.exists(self.iu_mesh_impression_path_split):
            os.mknod(self.iu_mesh_impression_path_split)
        json.dump(matched_split, open(self.iu_mesh_impression_path_split, 'w'))
        return matched_split

    def get_reports_by_exp(self, exp, split, r2gen_id, report):
        if split == 'train' and 4 < exp < 9:
            report += " <sep> " + self.iu_mesh_impression_split[split][r2gen_id]['impression']

        if split == 'train':
            if exp == 2:
                return report + " <sep>" + self.iu_mesh_impression_split[split][r2gen_id]['mesh']
            elif exp == 3:
                return report + " <sep>" + self.iu_mesh_impression_split[split][r2gen_id]['attr']
            elif exp == 4:
                return report + " <sep>" + self.iu_mesh_impression_split[split][r2gen_id]['mesh_attr']
            elif exp == 6:
                return report + self.iu_mesh_impression_split[split][r2gen_id]['mesh']
            elif exp == 7:
                return report + self.iu_mesh_impression_split[split][r2gen_id]['attr']
            elif exp == 8:
                return report + self.iu_mesh_impression_split[split][r2gen_id]['mesh_attr']
        return report

    def validate_association(self):
        self.analyze.get_number_of_normal()
        self.analyze.get_number_of_no_index()
        self.analyze.get_number_of_empty_mesh_asc()

        return (
                self.analyze.total_number_of_empty_mesh == self.analyze.total_number_of_normal + self.analyze.total_number_of_no_index
                and self.analyze.train_number_of_empty_mesh == self.analyze.train_number_of_normal + self.analyze.train_number_of_no_index
                and self.analyze.val_number_of_empty_mesh == self.analyze.val_number_of_normal + self.analyze.val_number_of_no_index
                and self.analyze.test_number_of_empty_mesh == self.analyze.test_number_of_normal + self.analyze.test_number_of_no_index)
    ############################################
