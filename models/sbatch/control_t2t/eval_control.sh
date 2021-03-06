#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=eval_t2t_control
#SBATCH --output=eval_t2t_control.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../../ts_model/run.py \
    --name t2t_control \
    --mode infer \
    --num_cpu 5 \
    --model_mode "t2t:bert_vocab" \
    --exp_dir "/zfs1/hdaqing/saz31/ts_exp/" \
    --train_batch_size 32 \
    --dimension 768 \
    --num_hidden_layers 6 \
    --num_heads 8 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --control_mode "length|0.5:syntax|0.0:split|1.0:ppdb|1.0" \
    --max_ppdb_len 100 \
    --beam_search_size 1 \
    --lr 0.1 \
    --num_ref 8 \
    --infer_ref_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune_refs/tune.8turkers.tok.turk." \
    --infer_tfexample "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.example" \
    --infer_src_file "/zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.8turkers.tok.norm.ori" \
    --eval_batch_size 100 \
    --bert_vocab_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt" \
    --ppdb_vocab "/zfs1/hdaqing/saz31/dataset/rule_v3/vocab" \
    --ppdb_file "/zfs1/hdaqing/saz31/dataset/ppdb.txt" \
    --bert_ckpt_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" \
    --bert_config_file "/ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/bert_config.json"