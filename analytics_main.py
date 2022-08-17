# exp setup
import time

import timer
from _global.argument_parser import ArgumentParser
from analytics.analyze import Analyze
from analytics.plot import Plot
from data_processors.data_processor import DataProcessor
import matplotlib.pyplot as plt


def main():
    start_time = time.time()
    # parse arguments
    args = ArgumentParser().args

    # r2gen split
    args.is_new_random_split = 0
    data_processor = DataProcessor(args)

    print("######### before split#########")
    data_processor.analyze.print_normal_percentage()
    data_processor.analyze.print_no_index_percentage()
    data_processor.analyze.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())

    plot = Plot(args, data_processor.analyze)
    # normal ratio
    plot.plot_duel_stacked_bar(num=1, x="split", ys=["t_ratio", "normal_ratio"],
                               colors=[plot.abnormal_color, plot.normal_color], labels=['Normal', 'Abnormal'])
    # indexed ratio
    plot.plot_duel_stacked_bar(num=2, x="split", ys=["t_ratio", "no_index_ratio"],
                               colors=[plot.indexed_color, plot.no_index_color], labels=['No Indexing', 'Indexed'])
    # empty mesh ratio
    plot.plot_duel_stacked_bar(num=3, x="split", ys=["t_ratio", "empty_mesh_ratio"],
                               colors=[plot.with_mesh_color, plot.empty_mesh_color], labels=['MeSH', 'No MeSH'])

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
    plot.plot_duel_stacked_bar(num=4, x="split", ys=["t_ratio", "normal_ratio"],
                               colors=[plot.abnormal_color, plot.normal_color], labels=['Normal', 'Abnormal'])
    # indexed ratio
    plot.plot_duel_stacked_bar(num=5, x="split", ys=["t_ratio", "no_index_ratio"],
                               colors=[plot.indexed_color, plot.no_index_color], labels=['No Indexing', 'Indexed'])
    # empty mesh ratio
    plot.plot_duel_stacked_bar(num=6, x="split", ys=["t_ratio", "empty_mesh_ratio"],
                               colors=[plot.with_mesh_color, plot.empty_mesh_color], labels=['MeSH', 'No MeSH'])
    plt.show()
    timer.time_executed(start_time, "R2Gen.Analysis")


if __name__ == '__main__':
    main()
#############################################################################################
