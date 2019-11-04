#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=t2t_eval_wiki_large
#SBATCH --output=t2t_eval_wiki_large.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=long


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../ts_model/run.py \
    --name plain_wiki_large \
    --mode infer \
    --num_cpu 5 \
    --model_mode t2t:bert_vocab \
    --exp_dir /zfs1/hdaqing/saz31/ts_exp/ \
    --train_batch_size 32 \
    --dimension 768 \
    --num_hidden_layers 12 \
    --num_heads 12 \
    --max_src_len 150 \
    --max_trg_len 150 \
    --beam_search_size 1 \
    --lr 0.01 \
    --num_ref 8 \
    --infer_ref_file /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune_refs/tune.8turkers.tok.turk. \
    --infer_tfexample /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.example \
    --infer_src_file /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.8turkers.tok.norm.ori \
    --eval_batch_size 100 \
    --bert_vocab_file /ihome/hdaqing/saz31/ts_2020/language_model/bert/uncased_L-12_H-768_A-12/vocab.txt \