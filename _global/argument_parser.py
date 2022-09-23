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
                             'to print <sep> and mesh annotation remove_annotation needs to be 1'
                        )
    # Remove annotation for evaluation
    parser.add_argument('--remove_annotation', type=int, default=1, choices=[0, 1],
                        help='0 to not remove annotation'
                             '1 to remove annotation'
                             'to print <sep> and mesh annotation remove_annotation needs to be 1'
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
    # Save graphs
    parser.add_argument('--is_save_plot', type=int, default=1, choices=[0, 1],
                        help='0 to not save generated plot'
                             '1 to save generated plot'
                        )
    # val and test whole dataset; only normal; only abnormal(with no indexing);
    parser.add_argument('--val_test_partial_data', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0 whole dataset' '296 590'
                             '1 only normal' '98 307'
                             '2 only no indexing' '4 25'
                             '3 only abnormal(without no indexing)' '194 258'
                             '4 only abnormal(with no indexing)' '198 283'
                        )

    # Experiment number
    parser.add_argument('--exp', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='exp is between 1-8'
                             '1. Reproduce the result for R2Gen.'
                             '2. Add `MeSH` and `attributes`'
                             '3. Add autogenerated `MeSH` and `attributes`'
                             '4. Train and Predict `impression` provided in the `IU-Xray` dataset from kaggle.'
                             '5. Add `MeSH` and `attributes` to 4.'
                             '6. Add autogenerated `MeSH` and `attributes` to 4'
                        )
    # dataloader settings
    parser.add_argument('--max_seq_length', type=int, default=60,
                        help='1. 60(paper) train max 162 mean 37 median 34 mode 33 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '2. train max 166 mean 43 median 38 mode 35 # val max 95 mean 36 median 33 mode 26 # test max 106 mean 33 median 30 mode 33'
                             '3. train max 0 mean 0 median 0 mode 0 # val max 0 mean 0 median 0 mode 0 # test max 0 mean 0 median 0 mode 0'
                             '4. train max 129 mean 10 median 6 mode 5 # val max 47 mean 9 median 5 mode 4 # test max 58 mean 8 median 4 mode 4'
                             '5. train max 147 mean 16 median 11 mode 6 # val max 47 mean 9 median 5 mode 4 # test max 58 mean 8 median 4 mode 4'
                             '6. train max 0 mean 0 median 0 mode 0 # val max 0 mean 0 median 0 mode 0 # test max 0 mean 0 median 0 mode 0')
    ###################################################################################################################
    args = parser.parse_args()
    return args


# exp setup
class ArgumentParser(object):
    def __init__(self):
        self.args = parse_agrs()
###########################################################################################################################################################################################
