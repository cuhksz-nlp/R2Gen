import csv
import json
import os


class DataProcessor(object):
    def __init__(self, args):
        self.r2gen_ann_path = args.ann_path
        self.kaggle_iu_reports_path = args.kaggle_iu_reports_path
        self.iu_mesh_impression_path = args.iu_mesh_impression_path
        self.iu_mesh_impression = dict()
        if os.path.exists(self.iu_mesh_impression_path):
            self.iu_mesh_impression = json.loads(open(self.iu_mesh_impression_path, 'r').read())
        else:
            os.mknod(self.iu_mesh_impression_path)
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
        matched = dict(train=[], val=[], test=[])
        for split, samples in r2gen_splits_ids_reports.items():
            for sample in samples:
                for r2gen_id, r2gen_report in sample.items():
                    uid = r2gen_id.split('_')[0].replace("CXR", "")
                    if uid in kaggle_uids_mesh_impression:
                        kaggle_report = kaggle_uids_mesh_impression[uid]["report"]
                        if r2gen_report == kaggle_report:
                            iu_mesh = kaggle_uids_mesh_impression[uid]["MeSH"]
                            mesh_attr_text = " <sep>"
                            mesh_text = " <sep>"
                            attr_text = " <sep>"
                            for mesh_info in iu_mesh.split(';'):
                                if '/' in mesh_info:
                                    mesh_attr = mesh_info.split('/')
                                    mesh_text += " <mesh:{}>".format(mesh_attr[0])
                                    attr_text += " <attr:{}>".format(mesh_attr[1])
                                    mesh_attr_text += " <mesh:{}> <attr:{}>".format(mesh_attr[0], mesh_attr[1])
                            matched_info = {
                                r2gen_id: {"iu_mesh": iu_mesh, "mesh": mesh_text,
                                           "attr": attr_text, "mesh_attr": mesh_attr_text,
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

        json.dump(matched, open(self.iu_mesh_impression_path, 'w'))
        return matched
