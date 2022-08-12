from dataclasses import dataclass


@dataclass
class PrcntgDataclass:
    train_normal_prcn: float
    val_normal_prcn: float
    test_normal_prcn: float
    dataset_normal_prcn: float
    train_no_index_prcn: float
    val_no_index_prcn: float
    test_no_index_prcn: float
    dataset_no_index_prcn: float