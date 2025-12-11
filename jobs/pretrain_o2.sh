#!/bin/bash
#SBATCH --account=zitnik_mz189              # associated Slurm account
#SBATCH --job-name=pretrain_%j              # assign job name
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH --cpus-per-task 16                  # request cores
#SBATCH -t 2-30:00                          # runtime in D-HH:MM format
#SBATCH -p gpu_quad	                        # partition to run in
#SBATCH --mem=32G                           # memory for all cores
#SBATCH --gres=gpu:l40s:1                   # number of GPUs
#SBATCH -o /home/an252/PROTON/data/slurm/pretrain_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /home/an252/PROTON/data/slurm/pretrain_%j.err   # file to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anoori@college.harvard.edu

# Set local environment directory variable
export PROJECT_DIR="/home/an252/PROTON"

# Change working directory
cd ${PROJECT_DIR}

# Load modules
module load gcc/14.2.0 cuda/12.8

# Activate environment
source /home/an252/PROTON/.venv/bin/activate

export neurokg__paths__test_set=42049

python -m src.cli train
