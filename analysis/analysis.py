import json
import pandas as pd


class Analysis(object):
    def __init__(self, args):
        self.iu_mesh_impression_path = args.iu_mesh_impression_path
        self.dataset_size = 0
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        self.total_number_of_normal = 0
        self.train_number_of_normal = 0
        self.val_number_of_normal = 0
        self.test_number_of_normal = 0

    def get_normal_sample_percentage(self):
        with open(self.iu_mesh_impression_path, "rt") as datasetFile:
            dataset = json.load(datasetFile)

        for split, samples in dataset.items():
            if split == "train":
                self.train_size = len(samples)
                self.train_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "val":
                self.val_size = len(samples)
                self.val_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "test":
                self.test_size = len(samples)
                self.test_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            self.dataset_size = self.train_size + self.val_size + self.test_size
            self.total_number_of_normal = self.train_number_of_normal + self.val_number_of_normal + self.test_number_of_normal
        train_normal_prcn = self.train_number_of_normal / self.train_size * 100
        val_normal_prcn = self.val_number_of_normal / self.val_size * 100
        test_normal_prcn = self.test_number_of_normal / self.test_size * 100
        dataset_normal_prcn = self.total_number_of_normal / self.dataset_size * 100
        return train_normal_prcn, val_normal_prcn, test_normal_prcn, dataset_normal_prcn
