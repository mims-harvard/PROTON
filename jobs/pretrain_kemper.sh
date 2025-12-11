#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab       # associated Slurm account
#SBATCH --job-name=disease_split_%j         # assign job name
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH -c 16                               # request cores
#SBATCH -t 3-00:00                          # runtime in D-HH:MM format
#SBATCH -p kempner_h100	                    # partition to run in
#SBATCH --mem=50G                           # memory for all cores
#SBATCH --gres=gpu:1                        # number of GPUs
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/an252/PROTON/data/slurm/disease_split_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/an252/PROTON/data/slurm/disease_split_%j.err   # file to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anoori@college.harvard.edu

# Set local environment directory variable
export PROJECT_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/an252/PROTON"

# Change working directory
cd ${PROJECT_DIR}

# Load modules
module load gcc/14.2.0 cuda/12.4.1-fasrc01

# Activate environment
source /n/holylfs06/LABS/mzitnik_lab/Users/an252/PROTON/.venv/bin/activate

export neurokg__test_set=39291

# Run script
python -m src.cli train
