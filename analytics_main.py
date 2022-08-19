# exp setup
import statistics
import time

import matplotlib.pyplot as plt

import timer
from _global.argument_parser import ArgumentParser
from analytics.plot import Plot
from data_processors.data_processor import DataProcessor
from modules.tokenizers import Tokenizer


def main():
    start_time = time.time()
    # parse arguments
    args = ArgumentParser().args

    # r2gen split
    args.is_new_random_split = 0
    data_processor = DataProcessor(args)

    # tokenizer = Tokenizer(args, data_processor)
    #
    # counts = {split: [
    #     len(data_processor.get_reports_by_exp(8, split, rid, tokenizer.clean_report(sample["impression"])).split())
    #     for rid, sample in split_sample.items()]
    #           for split, split_sample in data_processor.iu_mesh_impression_split.items()}
    # stats = {
    #     split: {"max": max(split_count), "mean": statistics.mean(split_count), "meidan": statistics.median(split_count),
    #             "mode": statistics.mode(split_count)} for split, split_count in counts.items()}
    # print(stats)

    print("######### before split#########")
    data_processor.analyze.print_normal_percentage()
    data_processor.analyze.print_no_index_percentage()
    data_processor.analyze.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())

    plot = Plot(args, data_processor.analyze)
    # normal ratio
    plot.plot_stacked_bar(num=1, xs=["t_ratio", "normal_ratio"], ys=["split"],
                          colors=[plot.abnormal_color, plot.normal_color], labels=['Abnormal', 'Normal'],
                          number_of_col_in_legend=2, plot_name="[R2gen Split]Normal to Abnormal ratio")
    # indexed ratio
    plot.plot_stacked_bar(num=2, xs=["t_ratio", "no_index_ratio"], ys=["split"],
                          colors=[plot.indexed_color, plot.no_index_color], labels=['Indexed', 'No Indexing'],
                          number_of_col_in_legend=2, plot_name="[R2gen Split]Indexed to No Indexing ratio")
    # empty mesh ratio
    plot.plot_stacked_bar(num=3, xs=["t_ratio", "no_mesh_ratio"], ys=["split"],
                          colors=[plot.mesh_color, plot.no_mesh_color], labels=['MeSH', 'No MeSH'],
                          number_of_col_in_legend=2, plot_name="[R2gen Split]Mesh to No MeSH ratio")

    # new split
    args.is_new_random_split = 1
    data_processor = DataProcessor(args)
    print("######### after split#########")
    data_processor.analyze.print_normal_percentage()
    data_processor.analyze.print_no_index_percentage()
    data_processor.analyze.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())

    plot = Plot(args, data_processor.analyze)
    # normal ratio
    plot.plot_stacked_bar(num=4, xs=["t_ratio", "normal_ratio"], ys=["split"],
                          colors=[plot.abnormal_color, plot.normal_color], labels=['Abnormal', 'Normal'],
                          number_of_col_in_legend=2, plot_name="[New Split]Normal to Abnormal ratio")
    # indexed ratio
    plot.plot_stacked_bar(num=5, xs=["t_ratio", "no_index_ratio"], ys=["split"],
                          colors=[plot.indexed_color, plot.no_index_color], labels=['Indexed', 'No Indexing'],
                          number_of_col_in_legend=2, plot_name="[New Split]Indexed to No Indexing ratio")
    # empty mesh ratio
    plot.plot_stacked_bar(num=6, xs=["t_ratio", "no_mesh_ratio"], ys=["split"],
                          colors=[plot.mesh_color, plot.no_mesh_color], labels=['MeSH', 'No MeSH'],
                          number_of_col_in_legend=2, plot_name="[New Split]MeSH to No MeSH ratio")
    plt.show()
    timer.time_executed(start_time, "R2Gen.Analysis")


if __name__ == '__main__':
    main()
#############################################################################################
