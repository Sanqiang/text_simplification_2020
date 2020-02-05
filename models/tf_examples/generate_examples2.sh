#!/usr/bin/env bash

#SBATCH --cluster=htc
#SBATCH --job-name=generate_examples
#SBATCH --output=generate_examples.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"

# Run the job
srun python generate_examples.py