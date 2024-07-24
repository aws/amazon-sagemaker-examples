#!/bin/bash
#SBATCH --output=logs/%x_%j.out  # Redirects outputs to file in current_dir/logs
#SBATCH --error=logs/%x_%j.out  # Redirects err to same file in current_dir/logs
#SBATCH --job-name=prep_nmt_data
#SBATCH --ntasks-per-node=1
#SBATCH -N 1

CONDA_ENV_PATH=${1:-"$CONDA_DEFAULT_ENV"}
if [[ -z "${CONDA_ENV_PATH}" ]]; then
    echo "Conda env not set, exiting"
fi

srun -l -D `pwd` conda run -p $CONDA_ENV_PATH --no-capture-output python data/prepare_nemo_megatron_dataset.py
