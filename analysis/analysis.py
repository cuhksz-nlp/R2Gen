import json

from prcntg_data_class import PrcntgDataclass


class Analysis(object):
    def __init__(self, args):
        self.iu_mesh_impression_path = args.iu_mesh_impression_path
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

    def open_association_file(self):
        with open(self.iu_mesh_impression_path, "rt") as datasetFile:
            dataset = json.load(datasetFile)
        return dataset

    def get_normal_sample_percentage(self):
        self.get_samples_size()
        self.get_number_of_normal()
        self.get_number_of_no_index()

        train_normal_prcn = self.train_number_of_normal / self.train_size * 100
        val_normal_prcn = self.val_number_of_normal / self.val_size * 100
        test_normal_prcn = self.test_number_of_normal / self.test_size * 100
        dataset_normal_prcn = self.total_number_of_normal / self.dataset_size * 100
        train_no_index_prcn = self.train_number_of_no_index / self.train_size * 100
        val_no_index_prcn = self.val_number_of_no_index / self.val_size * 100
        test_no_index_prcn = self.test_number_of_no_index / self.test_size * 100
        dataset_no_index_prcn = self.total_number_of_no_index / self.dataset_size * 100
        return PrcntgDataclass(train_normal_prcn, val_normal_prcn, test_normal_prcn, dataset_normal_prcn,
                               train_no_index_prcn, val_no_index_prcn, test_no_index_prcn, dataset_no_index_prcn)

    def get_samples_size(self):
        dataset = self.open_association_file()

        for split, samples in dataset.items():
            if split == "train":
                self.train_size = len(samples)
            if split == "val":
                self.val_size = len(samples)
            if split == "test":
                self.test_size = len(samples)

            self.dataset_size = self.train_size + self.val_size + self.test_size

    def get_number_of_normal(self):
        dataset = self.open_association_file()

        for split, samples in dataset.items():
            if split == "train":
                self.train_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "val":
                self.val_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])
            if split == "test":
                self.test_number_of_normal = len([s for s in samples.keys() if samples[s].get("iu_mesh") == "normal"])

            self.total_number_of_normal = \
                self.train_number_of_normal + self.val_number_of_normal + self.test_number_of_normal

    def get_number_of_no_index(self):
        dataset = self.open_association_file()

        for split, samples in dataset.items():
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

    def get_number_of_empty_mesh(self):
        dataset = self.open_association_file()

        for split, samples in dataset.items():
            if split == "train":
                self.train_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            if split == "val":
                self.val_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            if split == "test":
                self.test_number_of_empty_mesh = len([s for s in samples.keys() if samples[s].get("mesh") == ""])
            self.total_number_of_empty_mesh = \
                self.train_number_of_empty_mesh + self.val_number_of_empty_mesh + self.test_number_of_empty_mesh

    def validate_association(self):
        self.get_number_of_normal()
        self.get_number_of_no_index()
        self.get_number_of_empty_mesh()
        return self.total_number_of_empty_mesh == self.total_number_of_normal + self.total_number_of_no_index

# train_normal:  32.96278395360077 %  val_normal:  33.108108108108105 %  test_normal:  52.03389830508475 %  dataset_normal:  36.78510998307953 %
# train_no_index:  1.9816336394393428 %  val_no_index:  1.3513513513513513 %  test_no_index:  4.23728813559322 %  dataset_no_index:  2.3688663282571913 %
