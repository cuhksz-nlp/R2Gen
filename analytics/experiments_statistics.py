# exp setup
import json
import statistics


class ExperimentsStatistics(object):
    def __init__(self, tokenizer, exp):
        self.tokenizer = tokenizer
        self.split_count = self.get_slit_count_by_exp(exp)
        self.stats = self.get_exp_stats()

    def get_slit_count_by_exp(self, exp):
        if 4 < exp < 9:
            split_count = {split: [len(
                self.tokenizer.data_processor.get_reports_by_exp(
                    exp, split, rid, self.tokenizer.clean_report(sample["impression"])).split())
                for rid, sample in split_sample.items()]
                for split, split_sample in self.tokenizer.data_processor.iu_mesh_impression_split.items()}
        else:
            split_count = {split: [len(
                self.tokenizer.data_processor.get_reports_by_exp(
                    exp, split, sample["id"], self.tokenizer.clean_report(sample["report"])).split())
                for sample in split_sample]
                for split, split_sample in self.tokenizer.ann.items()}
        return split_count

    def get_exp_stats(self):
        stats = {
            split: {"max": max(split_count), "mean": statistics.mean(split_count),
                    "median": statistics.median(split_count),
                    "mode": statistics.mode(split_count)} for split, split_count in self.split_count.items()}

        return stats
