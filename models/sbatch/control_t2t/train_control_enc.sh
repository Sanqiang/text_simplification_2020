#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=titanx
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_t2t_control_enc
#SBATCH --output=train_t2t_control_enc.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=short
#SBATCH --mem=32g


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../../ts_model/run.py \
    --name t2t_control_enc \
    --mode train \
    --num_cpu 5 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_tfexample "/zfs1/hdaqing/saz31/dataset/example_v3/*.example" \
    --train_batch_size 32 \
    --dimension 768 \
    --num_hidden_layers 6 \
    --num_heads 8 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --control_mode "length:syntax:split:ppdb:encoder" \
    --max_ppdb_len 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v3/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json" \
    --init_ckpt_path "/zfs1/hdaqing/saz31/ts_exp_bk/t2t_control_enc/log/model.ckpt-806000" \