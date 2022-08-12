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
    prcntg_dataclass = analysis.get_normal_sample_percentage()
    print("train_normal: ", prcntg_dataclass.train_normal_prcn, "% ", "val_normal: ", prcntg_dataclass.val_normal_prcn,
          "% ", "test_normal: ", prcntg_dataclass.test_normal_prcn, "% ", "dataset_normal: ",
          prcntg_dataclass.dataset_normal_prcn, "% ")
    print("train_no_index: ", prcntg_dataclass.train_no_index_prcn, "% ", "val_no_index: ",
          prcntg_dataclass.val_no_index_prcn, "% ", "test_no_index: ", prcntg_dataclass.test_no_index_prcn, "% ",
          "dataset_no_index: ", prcntg_dataclass.dataset_no_index_prcn, "% ")
    timer.time_executed(start_time, "R2Gen")


if __name__ == '__main__':
    main()
