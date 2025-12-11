#!/bin/bash
#SBATCH --account=zitnik_mz189             # associated Slurm account
#SBATCH --job-name=split_data_%j           # assign job name
#SBATCH --ntasks-per-node=1
#SBATCH -c 1                                # request cores
#SBATCH -t 0-02:00                          # runtime in D-HH:MM format
#SBATCH -p gpu_quad	                        # partition to run in
#SBATCH --mem=32G                           # memory for all cores
#SBATCH --gres=gpu:1                        # request GPU
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/slurm/split_data_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/slurm/split_data_%j.err   # file to which STDERR will be written, including job ID (%j)

# Change working directory
cd /n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON

# Load modules
module load gcc/9.2.0 cuda/11.7 python/3.9.14

# Activate environment
source proton_env/bin/activate

# Run script
python -m src.cli split
