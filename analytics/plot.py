import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", color_codes=True)


class Plot(object):
    def __init__(self, args, analyze):
        self.is_save_plot = args.is_save_plot
        self.analyze = analyze
        self.normal_color = "#64B5CD"
        self.abnormal_color = "#4C72B0"
        self.indexed_color = "#CCB974"
        self.no_index_color = "#DD8452"
        self.mesh_color = "#8172B3"
        self.no_mesh_color = "#DA8BC3"
        self.dataset = self.get_info_dict()

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
                          "no_mesh": self.analyze.train_number_of_empty_mesh,
                          "sample_size": self.analyze.train_size, "t_ratio": 1})
        data_dict.append({"split": "val", "normal": self.analyze.val_number_of_normal,
                          "no_index": self.analyze.val_number_of_no_index,
                          "no_mesh": self.analyze.val_number_of_empty_mesh,
                          "sample_size": self.analyze.val_size, "t_ratio": 1})
        data_dict.append({"split": "test", "normal": self.analyze.test_number_of_normal,
                          "no_index": self.analyze.test_number_of_no_index,
                          "no_mesh": self.analyze.test_number_of_empty_mesh,
                          "sample_size": self.analyze.test_size, "t_ratio": 1})

        dataset = pd.DataFrame(data_dict)
        dataset.loc[3] = pd.Series(
            ["total", dataset["normal"].sum(), dataset["no_index"].sum(), dataset["no_mesh"].sum(),
             dataset["sample_size"].sum(), 1], index=data_dict[0].keys())
        dataset["normal_ratio"] = dataset["normal"] / dataset["sample_size"]
        dataset["no_index_ratio"] = dataset["no_index"] / dataset["sample_size"]
        dataset["no_mesh_ratio"] = dataset["no_mesh"] / dataset["sample_size"]

        return dataset

    def plot_stacked_bar(self, num, xs, ys, colors, labels, number_of_col_in_legend, plot_name):
        plt.figure(num=num)
        bar_legends = []
        if len(xs) == len(ys) == len(colors) == len(labels):
            for i in range(len(xs)):
                sns.barplot(x=xs[i], y=ys[i], data=self.dataset, color=colors[i])
                bar_legends.append(mpatches.Patch(color=colors[i], label=labels[i]))
        else:
            if len(xs) == 1 and len(ys) == len(colors) == len(labels):
                for i in range(len(ys)):
                    sns.barplot(x=xs[0], y=ys[i], data=self.dataset, color=colors[i])
                    bar_legends.append(mpatches.Patch(color=colors[i], label=labels[i]))
            if len(ys) == 1 and len(xs) == len(colors) == len(labels):
                for i in range(len(xs)):
                    sns.barplot(x=xs[i], y=ys[0], data=self.dataset, color=colors[i])
                    bar_legends.append(mpatches.Patch(color=colors[i], label=labels[i]))
        plt.legend(loc='upper right', bbox_to_anchor=(1.01, 1.11),
                   ncol=number_of_col_in_legend, fancybox=True, shadow=True, handles=bar_legends)
        plt.title(plot_name, pad=28)
        if self.is_save_plot:
            plt.savefig("plot_assets/" + plot_name.lower().replace(" ", "_") + ".png")
