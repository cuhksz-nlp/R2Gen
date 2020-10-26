import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='.')
    parser.add_argument('--threshold', type=int, default=3, help='.')
    parser.add_argument('--num_workers', type=int, default=2, help='.')
    parser.add_argument('--batch_size', type=int, default=16, help='.')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='.')
    parser.add_argument('--d_ff', type=int, default=512, help='.')
    parser.add_argument('--d_vf', type=int, default=2048, help='.')
    parser.add_argument('--num_heads', type=int, default=8, help='.')
    parser.add_argument('--num_layers', type=int, default=3, help='.')
    parser.add_argument('--dropout', type=float, default=0.1, help='.')
    parser.add_argument('--logit_layers', type=int, default=1, help='.')
    parser.add_argument('--bos_idx', type=int, default=0, help='.')
    parser.add_argument('--eos_idx', type=int, default=0, help='.')
    parser.add_argument('--pad_idx', type=int, default=0, help='.')
    parser.add_argument('--use_bn', type=int, default=0, help='.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='.')
    parser.add_argument('--beam_size', type=int, default=3, help='.')
    parser.add_argument('--temperature', type=float, default=1.0, help='.')
    parser.add_argument('--sample_n', type=int, default=1, help='.')
    parser.add_argument('--group_size', type=int, default=1, help='.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='.')
    parser.add_argument('--epochs', type=int, default=100, help='.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='.')
    parser.add_argument('--record_dir', type=str, default='records/', help='.')
    parser.add_argument('--save_period', type=int, default=1, help='.')
    parser.add_argument('--verbosity', type=int, default=2, help='.')
    parser.add_argument('--monitor_mode', type=str, default='max', help='.', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='.')
    parser.add_argument('--early_stop', type=int, default=50, help='.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='.')
    parser.add_argument('--step_size', type=int, default=50, help='.')
    parser.add_argument('--gamma', type=float, default=0.1, help='.')
    parser.add_argument('--resume', type=str, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=123456, help='.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
