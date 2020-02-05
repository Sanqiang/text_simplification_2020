#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=titanx
#SBATCH --gres=gpu:1
#SBATCH --job-name=gpt2_117_eval
#SBATCH --output=gpt2_177_eval.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=long


# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python ../ts_model/run.py \
    --name gpt2_117 \
    --mode infer \
    --num_cpu 5 \
    --model_mode gpt2:gpt2_vocab \
    --exp_dir /zfs1/hdaqing/saz31/ts_exp/ \
    --train_batch_size 32 \
    --max_src_len 100 \
    --max_trg_len 100 \
    --beam_search_size 1 \
    --lr 0.01 \
    --num_ref 8 \
    --infer_ref_file /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune_refs/tune.8turkers.tok.turk. \
    --infer_tfexample /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.example \
    --infer_src_file /zfs1/hdaqing/saz31/dataset/tmp_wikilarge_2048/dev/tune.8turkers.tok.norm.ori \
    --eval_batch_size 100 \
    --gpt2_ckpt_path /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/117M/model.ckpt \
    --models_dir /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/ \
    --gpt2_ckpt_path /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/117M/model.ckpt \
    --models_dir /ihome/hdaqing/saz31/ts_2020/language_model/gpt2/models/ \
    --model_name 117M \
