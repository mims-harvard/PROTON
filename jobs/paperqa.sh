#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab       # associated Slurm account
#SBATCH --job-name=submit_PQA_%j            # assign job name
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH -c 16                               # request cores
#SBATCH -t 1-00:00                          # runtime in D-HH:MM format
#SBATCH -p kempner	                        # partition to run in
#SBATCH --mem=32G                           # memory for all cores
#SBATCH --gres=gpu:1                        # number of gpus
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON/data/slurm/query_PQA_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON/data/slurm/query_PQA_%j.err   # file to which STDERR will be written, including job ID (%j)

# Set local environment directory variable
export PROJECT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON"

# Change working directory
cd ${PROJECT_DIR}

# Load modules
module load gcc/13.2.0-fasrc01 cuda/11.8.0-fasrc01 python/3.12.5-fasrc01

# Activate environment
conda deactivate
source /n/home13/anoori/venvs/pqa_env/bin/activate

# Run script
python -m src.cli pqa submit --pqa-random
