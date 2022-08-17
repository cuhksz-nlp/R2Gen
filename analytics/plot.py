import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

sns.set_theme(style="ticks", color_codes=True)


# palette = ["#273eff", "#f37c04", "#4bc938", "#e82007", "#8b2be2", "#9f4700", "#f24cc1", "#a3a3a3", "#f7c401",
#                "#56d8fe", "#cd88a8", "#5ea55a", "#f15357", "#2d70b1", "#000200"]

class Plot(object):
    def __init__(self, args, analyze):
        self.args = args
        self.analyze = analyze
        self.normal_color = "#4eabb6"
        self.abnormal_color = "#b6a8eb"
        self.indexed_color = "#4bc938"
        self.no_index_color = "#e82007"
        self.with_mesh_color = "#8b2be2"
        self.empty_mesh_color = "#9f4700"
        self.dataset = self.get_info_dict()
        # self.normal_abnormal = self.get_normal_abnormal()
        # self.indexed_no_index = self.get_indexed_no_index()
        # self.with_mesh_empty_mesh = self.get_with_mesh_empty_mesh()

    def populate_analyze(self):
        self.analyze.get_number_of_normal()
        self.analyze.get_number_of_no_index()
        self.analyze.get_number_of_empty_mesh_asc()
        self.analyze.get_samples_size()

    def get_info_dict(self):
        self.populate_analyze()
        data_dict = list(dict())
        data_dict.append({"split": "train", "normal": self.analyze.train_number_of_normal,
                          "no_index": self.analyze.train_number_of_no_index,
                          "empty_mesh": self.analyze.train_number_of_empty_mesh,
                          "sample_size": self.analyze.train_size, "t_ratio": 1})
        data_dict.append({"split": "val", "normal": self.analyze.val_number_of_normal,
                          "no_index": self.analyze.val_number_of_no_index,
                          "empty_mesh": self.analyze.val_number_of_empty_mesh,
                          "sample_size": self.analyze.val_size, "t_ratio": 1})
        data_dict.append({"split": "test", "normal": self.analyze.test_number_of_normal,
                          "no_index": self.analyze.test_number_of_no_index,
                          "empty_mesh": self.analyze.test_number_of_empty_mesh,
                          "sample_size": self.analyze.test_size, "t_ratio": 1})

        dataset = pd.DataFrame(data_dict)
        dataset["normal_ratio"] = dataset["normal"] / dataset["sample_size"]
        dataset["no_index_ratio"] = dataset["no_index"] / dataset["sample_size"]
        dataset["empty_mesh_ratio"] = dataset["empty_mesh"] / dataset["sample_size"]

        return dataset

    def plot_duel_stacked_bar(self, num, x, ys, colors, labels):
        plt.figure(num=num)
        sns.barplot(x=x, y=ys[0], data=self.dataset, estimator=sum, ci=None, color=colors[0])
        sns.barplot(x=x, y=ys[1], data=self.dataset, color=colors[1])

        top_bar = mpatches.Patch(color=colors[1], label=labels[0])
        bottom_bar = mpatches.Patch(color=colors[0], label=labels[1])
        plt.legend(handles=[top_bar, bottom_bar])
