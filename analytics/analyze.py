# exp setup
import json


class Analyze(object):
    def __init__(self, args):
        self.iu_mesh_impression_path_split = args.iu_mesh_impression_path.replace(".json", "_split.json")
        # size
        self.dataset_size = 0
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        # Normal
        self.total_number_of_normal = 0
        self.train_number_of_normal = 0
        self.val_number_of_normal = 0
        self.test_number_of_normal = 0
        # No Indexing
        self.total_number_of_no_index = 0
        self.train_number_of_no_index = 0
        self.val_number_of_no_index = 0
        self.test_number_of_no_index = 0
        # empty mesh
        self.total_number_of_empty_mesh = 0
        self.train_number_of_empty_mesh = 0
        self.val_number_of_empty_mesh = 0
        self.test_number_of_empty_mesh = 0

        with open(self.iu_mesh_impression_path_split, "rt") as datasetFile:
            self.dataset = json.load(datasetFile)

    def get_normal_ratio(self):
        self.get_samples_size()
        self.get_number_of_normal()

        train_normal_prcn = self.train_number_of_normal / self.train_size
        val_normal_prcn = self.val_number_of_normal / self.val_size
        test_normal_prcn = self.test_number_of_normal / self.test_size
        dataset_normal_prcn = self.total_number_of_normal / self.dataset_size
        return train_normal_prcn, val_normal_prcn, test_normal_prcn, dataset_normal_prcn

    def print_normal_percentage(self):
        normal_prcn_tuple = self.get_normal_ratio()
        print("train_normal: ", normal_prcn_tuple[0] * 100, "% ", "val_normal: ", normal_prcn_tuple[1] * 100, "% ",
              "test_normal: ", normal_prcn_tuple[2] * 100, "% ", "dataset_normal: ", normal_prcn_tuple[3] * 100, "% ")

    def get_no_index_ratio(self):
        self.get_samples_size()
        self.get_number_of_no_index()

        train_no_index_prcn = self.train_number_of_no_index / self.train_size
        val_no_index_prcn = self.val_number_of_no_index / self.val_size
        test_no_index_prcn = self.test_number_of_no_index / self.test_size
        dataset_no_index_prcn = self.total_number_of_no_index / self.dataset_size
        return train_no_index_prcn, val_no_index_prcn, test_no_index_prcn, dataset_no_index_prcn

    def print_no_index_percentage(self):
        no_index_prcn_tuple = self.get_no_index_ratio()
        print("train_no_index: ", no_index_prcn_tuple[0] * 100, "% ", "val_no_index: ", no_index_prcn_tuple[1] * 100,
              "% ", "test_no_index: ", no_index_prcn_tuple[2] * 100, "% ", "dataset_no_index: ",
              no_index_prcn_tuple[3] * 100, "% ")

    def get_empty_mesh_asc_ratio(self):
        self.get_samples_size()
        self.get_number_of_empty_mesh_asc()

        train_empty_mesh_prcn = self.train_number_of_empty_mesh / self.train_size
        val_empty_mesh_prcn = self.val_number_of_empty_mesh / self.val_size
        test_empty_mesh_prcn = self.test_number_of_empty_mesh / self.test_size
        dataset_empty_mesh_prcn = self.total_number_of_empty_mesh / self.dataset_size
        return train_empty_mesh_prcn, val_empty_mesh_prcn, test_empty_mesh_prcn, dataset_empty_mesh_prcn

    def print_empty_mesh_asc_percentage(self):
        empty_mesh_prcn_tuple = self.get_empty_mesh_asc_ratio()
        print("train_empty_mesh: ", empty_mesh_prcn_tuple[0] * 100, "% ", "val_empty_mesh: ",
              empty_mesh_prcn_tuple[1] * 100, "% ", "test_empty_mesh: ", empty_mesh_prcn_tuple[2] * 100, "% ",
              "dataset_empty_mesh: ", empty_mesh_prcn_tuple[3] * 100, "% ")

    def get_samples_size(self):
        for split, samples in self.dataset.items():
            if split == "train":
                self.train_size = len(samples)
            if split == "val":
                self.val_size = len(samples)
            if split == "test":
                self.test_size = len(samples)

            self.dataset_size = self.train_size + self.val_size + self.test_size

    def get_number_of_normal(self):
        for split, samples in self.dataset.items():
            if split == "train":
                self.train_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "val":
                self.val_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "test":
                self.test_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])

            self.total_number_of_normal = \
                self.train_number_of_normal + self.val_number_of_normal + self.test_number_of_normal

    def get_number_of_no_index(self):
        for split, samples in self.dataset.items():
            if split == "train":
                self.train_number_of_no_index = len(
                    [s for s in samples.keys() if samples[s].get("iu_mesh") == "No Indexing"])
            if split == "val":
                self.val_number_of_no_index = len(
                    [s for s in samples.keys() if samples[s].get("iu_mesh") == "No Indexing"])
            if split == "test":
                self.test_number_of_no_index = len(
                    [s for s in samples.keys() if samples[s].get("iu_mesh") == "No Indexing"])
            self.total_number_of_no_index = \
                self.train_number_of_no_index + self.val_number_of_no_index + self.test_number_of_no_index

    def get_number_of_empty_mesh_asc(self):
        for split, samples in self.dataset.items():
            if split == "train":
                self.train_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            if split == "val":
                self.val_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            if split == "test":
                self.test_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            self.total_number_of_empty_mesh = \
                self.train_number_of_empty_mesh + self.val_number_of_empty_mesh + self.test_number_of_empty_mesh

################################################################################################################################