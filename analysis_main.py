import argparse
import time

import timer
from analytics.analyze import Analyze
from data_processors.data_processor import DataProcessor


def parse_agrs():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--ann_path', type=str, default='../../data/iu_xray/r2gen/annotation.json',
                        help='the path to the directory containing the data.')
    # Reports path kaggle iu xray
    parser.add_argument('--kaggle_iu_projections_path', type=str,
                        default='../../data/iu_xray/kaggle/iu_projections.csv',
                        help='the path to the directory containing the projections data.')
    parser.add_argument('--kaggle_iu_reports_path', type=str, default='../../data/iu_xray/kaggle/iu_reports.csv',
                        help='the path to the directory containing the reports.')
    parser.add_argument('--iu_mesh_impression_path', type=str,
                        default='../../data/iu_xray/kaggle/iu_mesh_impression.json',
                        help='the path to the directory containing the mesh and impression for r2gen dataset.')
    # Create iu_mesh_impression.json
    parser.add_argument('--create_r2gen_kaggle_association', type=int, default=1,
                        help='0 to not create association'
                             '1 to create association'
                        )
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    # parse arguments
    args = parse_agrs()

    # Process data to get additional info
    data_processor = DataProcessor(args)

    # analytics
    analysis = Analyze(args)
    analysis.print_normal_percentage()
    analysis.print_no_index_percentage()
    analysis.print_empty_mesh_asc_percentage()
    print("Is association file valid: ", analysis.validate_association())
    timer.time_executed(start_time, "R2Gen.Analysis")


if __name__ == '__main__':
    main()
