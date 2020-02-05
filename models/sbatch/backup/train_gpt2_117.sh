#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpt2_117_train
#SBATCH --output=gpt2_117_train.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=40g


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../ts_model/run.py \
    --name gpt2_117 \
    --mode train \
    --num_cpu 5 \
    --model_mode gpt2:gpt2_vocab \
    --exp_dir /zfs1/hdaqing/saz31/ts_exp/ \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/example/*" \
    --train_batch_size 32 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --eval_batch_size 64 \
    --gpt2_ckpt_path /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/117M/model.ckpt \
    --models_dir /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/ \
    --gpt2_ckpt_path /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/117M/model.ckpt \
    --models_dir /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/ \
    --model_name 117M \
