#!/bin/bash
set -xe

export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/versions/miniconda3-latest/envs/r2gen/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


pyenv local miniconda3-latest

cd "$HOME/projects/R2Gen"

time python main.py \
--image_dir ../data/iu_xray/images/ \
--ann_path ../data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 90 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223

