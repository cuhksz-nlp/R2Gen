# exp setup
import time

import timer
from _global.argument_parser import ArgumentParser
from analytics.analyze import Analyze
from analytics.plot import Plot
from data_processors.data_processor import DataProcessor


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

    # new split
    args.is_new_random_split = 1
    data_processor = DataProcessor(args)
    print("######### after split#########")
    data_processor.analyze.print_normal_percentage()
    data_processor.analyze.print_no_index_percentage()
    data_processor.analyze.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())
    # plot = Plot(args)
    # plot.plot_bar()
    timer.time_executed(start_time, "R2Gen.Analysis")


if __name__ == '__main__':
    main()
#############################################################################################