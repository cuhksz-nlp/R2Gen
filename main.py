import torch
import argparse
import numpy as np
import os
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from modules.utilities import copy_checkpoint, evaluate, store, merge, remove_temporary_folders
from modules.image_processor import image_preprocessor
from models.r2gen import R2GenModel
from models.text_embedding import TextEmbeddingModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args


def main():
    
    # our arguments
    data_src = 'iu_xray'     # ['iu_xray', 'standardized_iu_xray']
    whether_to_train = True  # [True, False]
    api_key = 'empty'        # ['empty', '<your_api_key>']

    # additional arguments
    whether_to_generate_r2gen_result = True  # [True, False]
    dataset_type = ['test']                  # ['train', 'val', 'test']

    # parse arguments
    args = parse_agrs()
    args.image_dir = 'data/' + data_src + '/images'
    args.ann_path = 'data/' + data_src + '/annotation.json'
    args.dataset_name = data_src
    args.save_dir = 'results/' + data_src
    if whether_to_train == False:
        args.resume = 'output/pth/model_' + data_src + '.pth'

    # preprocess images
    if data_src == 'standardized_iu_xray':
        if not os.path.exists('data/standardized_iu_xray'):
            image_preprocessor()

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
    if whether_to_train == True:
        trainer.train()
        copy_checkpoint(data_src)

    # generate r2gen model result
    if whether_to_generate_r2gen_result == True:
        r2gen_result = trainer.predict(dataset_type)
        store(r2gen_result, 'output/r2gen_result', data_src + '_result.json')

    # compute r2gen model score
    if whether_to_generate_r2gen_result == True:
        r2gen_score = evaluate(r2gen_result, dataset_type)
        store(r2gen_score, 'output/r2gen_score', data_src + '_score.json')

    # bulid text embedding model
    if api_key != 'empty':
        ann_path = 'data/mimic_cxr/annotation.json'
        data_path = 'output/r2gen_result/' + data_src + '_result.json'
        record_dir = 'output/record'
        record_file = data_src + '_record.json'
        text_embedding_model = TextEmbeddingModel(ann_path, data_path, record_dir, record_file, api_key)

    # generate text embedding model result
    if api_key != 'empty':
        text_embedding_result = text_embedding_model.refine(dataset_type)
        store(text_embedding_result, 'output/text_embedding_result', data_src + '_result.json')

    # compute text embedding model score
    if api_key != 'empty':
        text_embedding_score = evaluate(text_embedding_result, dataset_type)
        store(text_embedding_score, 'output/text_embedding_score', data_src + '_score.json')

    # merge result
    if api_key != 'empty':
        r2gen_result_path = 'output/r2gen_result/' + data_src + '_result.json'
        record_path = 'output/record/' + data_src + '_record.json'
        union_result_dir_name = 'output/union_result'
        union_result_file_name = data_src + '_result.txt'
        merge(r2gen_result_path, record_path, union_result_dir_name, union_result_file_name)
    
    # remove temporary folders
    remove_temporary_folders(data_src)


if __name__ == '__main__':
    main()
