# exp setup
import time

import timer
from _global.argument_parser import ArgumentParser
from analytics.analyze import Analyze
from data_processors.data_processor import DataProcessor


def main():
    start_time = time.time()
    # parse arguments
    args = ArgumentParser().args

    # Process data to get additional info
    data_processor = DataProcessor(args)
    # analytics
    analysis = Analyze(args)
    print("######### before split#########")
    analysis.print_normal_percentage()
    analysis.print_no_index_percentage()
    analysis.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())
    data_processor.split_dataset()
    analysis = Analyze(args)
    print("######### after split#########")
    analysis.print_normal_percentage()
    analysis.print_no_index_percentage()
    analysis.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", data_processor.validate_association())
    timer.time_executed(start_time, "R2Gen.Analysis")


if __name__ == '__main__':
    main()
#############################################################################################