#!/usr/bin/bash
#SBATCH --job-name=kndvichallenge
#SBATCH --output=logs/kndvi_%j.out
#SBATCH --error=logs/kndvi_%j.err
#SBATCH --partition=clara
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Ensure log directory exists for SLURM to write output
mkdir -p logs
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Not running on SLURM, set up the environment"
    DATA_DIR="../kamay_ndvi_prediction/data"
else
    echo "Running on SLURM"
    module load Anaconda3/2024.02-1
    # TODO: put data on sc
    DATA_DIR="kndvi-prediction-challenge-baseline-submission-25"
fi

CONDA_ENV_NAME="time_gpt_env"

# set up the environment
# test if environment already exists
if conda env list | grep -q $CONDA_ENV_NAME; then
    echo "using existing environment"
    conda env update --file=environment.yml --solver=libmamba
else
    echo "creating new environment"
    conda env create -y --file=environment.yml --solver=libmamba
fi

# for whatever reason this is required to be able to activate envs
 . $(dirname $CONDA_PYTHON_EXE)/activate
conda activate $CONDA_ENV_NAME

which python

#### Run script
RUN_CMD="python code/orchestrator.py $DATA_DIR"
if [ -z "$SLURM_JOB_ID" ]; then
    $RUN_CMD
else
    srun $RUN_CMD
fi

