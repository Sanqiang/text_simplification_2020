#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=titanx
#SBATCH --gres=gpu:1
#SBATCH --job-name=t2t_train
#SBATCH --output=t2t_train.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../ts_model/run.py \
    --name plain \
    --mode train \
    --num_cpu 5 \
    --model_mode t2t:bert_vocab \
    --exp_dir /zfs1/hdaqing/saz31/ts_exp/ \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/example/*.example" \
    --train_batch_size 32 \
    --dimension 256 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --bert_vocab_file /ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt \
    --gpt2_ckpt_path /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/117M/model.ckpt \
    --models_dir /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/ \
    --model_name 117M \
