#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_split_large
#SBATCH --output=train_csplit_large.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../../ts_model/run.py \
    --name split_large \
    --mode train \
    --num_cpu 5 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikisplit_8192/example/*.example" \
    --train_batch_size 32 \
    --dimension 768 \
    --num_hidden_layers 12 \
    --num_heads 12 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --control_mode "" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v1/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt"