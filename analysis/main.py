import argparse
import time

import timer
from analysis import Analysis


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iu_mesh_impression_path', type=str,
                        default='../../data/iu_xray/kaggle/iu_mesh_impression.json',
                        help='the path to the directory containing the mesh and impression for r2gen dataset.')
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    # parse arguments
    args = parse_agrs()
    analysis = Analysis(args)
    train_normal_prcn, val_normal_prcn, test_normal_prcn, dataset_normal_prcn = analysis.get_normal_sample_percentage()
    print("train_normal: ", train_normal_prcn, "% ", "val_normal: ", val_normal_prcn, "% ", "test_normal: ",
          test_normal_prcn, "% ", "dataset_normal: ", dataset_normal_prcn, "% ")
    timer.time_executed(start_time, "R2Gen")


if __name__ == '__main__':
    main()
