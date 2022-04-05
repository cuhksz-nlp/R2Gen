#!/bin/bash
set -xe

export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/versions/miniconda3-latest/envs/r2gen/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


pyenv local miniconda3-latest

cd "$HOME/projects/R2Gen"

python main.py \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 30 \
--save_dir results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--seed 456789
