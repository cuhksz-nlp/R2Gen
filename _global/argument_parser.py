import argparse


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

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
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
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
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    # exp setup
    # Data input settings
    data_iu_xray_path = 'data/iu_xray'
    parser.add_argument('--image_dir', type=str, default='../data/iu_xray/r2gen/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default=('%s/r2gen/annotation.json' % data_iu_xray_path),
                        help='the path to the directory containing the data.')

    # Reports path kaggle iu xray
    parser.add_argument('--kaggle_iu_projections_path', type=str,
                        default=('%s/kaggle/iu_projections.csv' % data_iu_xray_path),
                        help='the path to the directory containing the projections data.')
    parser.add_argument('--kaggle_iu_reports_path', type=str,
                        default=('%s/kaggle/iu_reports.csv' % data_iu_xray_path),
                        help='the path to the directory containing the reports.')
    # Path for the generated association file
    parser.add_argument('--iu_mesh_impression_path', type=str,
                        default=('%s/kaggle/iu_mesh_impression.json' % data_iu_xray_path),
                        help='the path to the directory containing the mesh and impression for r2gen dataset.')

    # To print and debug
    parser.add_argument('--is_print', type=int, default=0, choices=[0, 1],
                        help='0 to not print the validation and test output with ground truth'
                             '1 to print the validation and test output with ground truth'
                        )
    # Remove annotation for evaluation
    parser.add_argument('--remove_annotation', type=int, default=1, choices=[0, 1],
                        help='0 to not remove annotation'
                             '1 to remove annotation'
                        )
    # specify sample size for train, val and test
    parser.add_argument('--train_sample', type=int, default=0, help='number of sample for training dataset')
    parser.add_argument('--val_sample', type=int, default=0, help='number of sample for validation dataset')
    parser.add_argument('--test_sample', type=int, default=0, help='number of sample for test dataset')

    # Create iu_mesh_impression.json
    parser.add_argument('--create_r2gen_kaggle_association', type=int, default=1, choices=[0, 1],
                        help='0 to not create association'
                             '1 to create association'
                        )
    # Create new random split to kaggle/annotation.json
    parser.add_argument('--is_new_random_split', type=int, default=1, choices=[0, 1],
                        help='0 to not create new random split'
                             '1 to create new random split'
                        )

    # Experiment number
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='exp is between 1-8 '
                             '1. Reproduce the result for R2Gen. '
                             '2. Add `MeSH` information provided in the `IU-Xray` dataset from kaggle. '
                             '3. Add `attributes` provided in the `IU-Xray` dataset from kaggle. '
                             '4. Add `MeSH` and `attributes` both. '
                             '5. Add `impression` provided in the `IU-Xray` dataset from kaggle. '
                             '6. Add `MeSH` and `impression`. '
                             '7. Add `attributes` and `impression`. '
                             '8. Add `MeSH`, `attributes` and `impression`.')
    # dataloader settings
    parser.add_argument('--max_seq_length', type=int, default=60, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='1. 60(paper) train max 162 mean 37 median 34 mode 33 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '2. train max 164 mean 39 median 35 mode 34 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '3. train max 164 mean 39 median 35 mode 34 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '4. train max 165 mean 41 median 37 mode 33 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '5. train max 191 mean 45 median 40 mode 38 # val max 114 mean 44 median 40 mode 50 # test max 154 mean 40 median 35 mode 38'
                             '6. train max 197 mean 47 median 42 mode 38 # val max 114 mean 44 median 40 mode 50 # test max 154 mean 40 median 35 mode 38'
                             '7. train max 197 mean 47 median 42 mode 38 # val max 114 mean 44 median 40 mode 50 # test max 154 mean 40 median 35 mode 38'
                             '8. train max 202 mean 49 median 43 mode 38 # val max 114 mean 44 median 40 mode 50 # test max 154 mean 40 median 35 mode 38')
    ###################################################################################################################
    args = parser.parse_args()
    return args


# exp setup
class ArgumentParser(object):
    def __init__(self):
        self.args = parse_agrs()
###########################################################################################################################################################################################
