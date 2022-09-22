#!/bin/bash
set -xe

export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/versions/miniconda3-latest/envs/r2gen/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


pyenv local miniconda3-latest

cd "$HOME/projects/R2Gen"

time python main.py \
--image_dir ../data/iu_xray/r2gen/images/ \
--dataset_name iu_xray \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--ann_path data/iu_xray/r2gen/annotation.json \
--save_dir ../r2gen_results/iu_xray \
--iu_mesh_impression_path data/iu_xray/kaggle/iu_mesh_impression.json \
--is_print 1 \
--remove_annotation 1 \
--train_sample 0 \
--val_sample 0 \
--test_sample 0 \
--create_r2gen_kaggle_association 0 \
--is_new_random_split 0 \
--max_seq_length 90 \
--exp 2