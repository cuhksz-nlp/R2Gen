import csv
import json
import os


class DataProcessor(object):
    # exp setup
    def __init__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.iu_mesh_impression_path = args.iu_mesh_impression_path
        self.iu_mesh_impression = dict()
        if os.path.exists(self.iu_mesh_impression_path):
            self.iu_mesh_impression = json.loads(open(self.iu_mesh_impression_path, 'r').read())
        else:
            self.iu_mesh_impression = self.associate_iu_r2gen_kaggle_by_id()

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

        unmatched = dict(train=[], val=[], test=[])
        matched = dict(train={}, val={}, test={})
        for split, samples in r2gen_splits_ids_reports.items():
            for sample in samples:
                for r2gen_id, r2gen_report in sample.items():
                    uid = r2gen_id.split('_')[0].replace("CXR", "")
                    if uid in kaggle_uids_mesh_impression:
                        kaggle_report = kaggle_uids_mesh_impression[uid]["report"]
                        if r2gen_report == kaggle_report:
                            iu_mesh = kaggle_uids_mesh_impression[uid]["MeSH"]
                            mesh_text = " <sep>"
                            attr_text = " <sep>"
                            mesh_attr_text = " <sep>"
                            for mesh_info in iu_mesh.split(';'):
                                if '/' in mesh_info:
                                    mesh_attr = mesh_info.split('/')
                                    ma_text = ""
                                    if ',' in mesh_attr[0]:
                                        for ma in mesh_attr[0].split(','):
                                            ma_text += " <mesh:{}>".format(ma.strip().replace(' ', '_'))
                                    else:
                                        ma_text = " <mesh:{}>".format(mesh_attr[0].strip().replace(' ', '_'))
                                    mesh_text += ma_text
                                    attr_text += " <attr:{}>".format(mesh_attr[1].strip().replace(' ', '_'))
                                    mesh_attr_text += "{} <attr:{}>".format(ma_text, mesh_attr[1].strip().replace(' ', '_'))
                            if " <sep>" == mesh_attr_text:
                                matched[split][r2gen_id] = {
                                    "iu_mesh": iu_mesh, "mesh": "", "attr": "",
                                    "mesh_attr": "",
                                    "impression": kaggle_uids_mesh_impression[uid]["impression"]}
                            else:
                                matched[split][r2gen_id] = {
                                    "iu_mesh": iu_mesh, "mesh": mesh_text, "attr": attr_text,
                                    "mesh_attr": mesh_attr_text,
                                    "impression": kaggle_uids_mesh_impression[uid]["impression"]}
                            # matched[split].append(matched_info)
                        else:
                            unmatched[split][r2gen_id] = {"r2gen_uid": uid, "r2gen_report": r2gen_report,
                                                          "kaggle_report": kaggle_report}
                            # unmatched[split].append(unmatched_info)
                    else:
                        unmatched_info = {
                            r2gen_id: {"r2gen_uid": uid, "r2gen_report": r2gen_report, "kaggle_report": ""}}
                        unmatched[split].append(unmatched_info)
        os.mknod(self.iu_mesh_impression_path)
        json.dump(matched, open(self.iu_mesh_impression_path, 'w'))
        return matched

    def get_reports_by_exp(self, exp, split, r2gen_id, report):
        if exp == 1:
            return report
        # only if split is train except for impression
        elif 4 < exp < 9:  # impression needs to be cleaned
            report += self.iu_mesh_impression[split][r2gen_id]['impression']
        if split == 'train':
            if exp == 2:
                return report + self.iu_mesh_impression[split][r2gen_id]['mesh']
            elif exp == 3:
                return report + self.iu_mesh_impression[split][r2gen_id]['attr']
            elif exp == 4:
                return report + self.iu_mesh_impression[split][r2gen_id]['mesh_attr']
            elif exp == 5:
                return report
            elif exp == 6:
                return report + self.iu_mesh_impression[split][r2gen_id]['mesh']
            elif exp == 7:
                return report + self.iu_mesh_impression[split][r2gen_id]['attr']
            elif exp == 8:
                return report + self.iu_mesh_impression[split][r2gen_id]['mesh_attr']
    ############################################
