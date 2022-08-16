import time

import numpy as np
import torch

import timer
from _global.argument_parser import ArgumentParser
from data_processors.data_processor import DataProcessor
from models.r2gen import R2GenModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer


def main():
    start_time = time.time()
    # parse arguments
    args = ArgumentParser().args

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # exp setup
    # Process data to get additional info
    data_processor = DataProcessor(args)

    if data_processor.validate_association():
        raise Exception("Association file is not valid")

    # create tokenizer
    tokenizer = Tokenizer(args, data_processor)
    ####################################

    # create data loader
    train_dataloader = R2DataLoader(args, data_processor, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, data_processor, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, data_processor, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()
    timer.time_executed(start_time, "R2Gen")


if __name__ == '__main__':
    main()
