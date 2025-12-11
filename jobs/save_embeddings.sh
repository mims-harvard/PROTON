#!/bin/bash
#SBATCH --account=zitnik_mz189              # associated Slurm account
#SBATCH --job-name=save_embeddings_%j       # assign job name
#SBATCH --ntasks-per-node=1                 # number of tasks per node
#SBATCH -c 4                                # request cores
#SBATCH -t 0-01:00                          # runtime in D-HH:MM format
#SBATCH -p highmem	                        # partition to run in
#SBATCH --mem=400G                          # memory for all cores
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/slurm/save_embeddings_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/slurm/save_embeddings_%j.err   # file to which STDERR will be written, including job ID (%j)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anoori@college.harvard.edu

# For saving full embeddings, run for 3-0:00 in medium partition with 200 GBs RAM
# For saving decoder, run for 0-01:00 in short partition with 100GBs RAM
# For saving full embeddings with high memory, run for 5-0:00 in highmem partition with 700 GBs RAM

# Change working directory
export PROJECT_DIR="/n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data"
export SPLIT_FOLDER='pretraining'
export BEST_CKPT='2025-03-02_ca053b65_epoch=2-step=69062.ckpt'
cd ${PROJECT_DIR}

# Load modules
module load gcc/9.2.0 cuda/11.7 python/3.9.14

# Activate environment
source proton_env/bin/activate

# Save embeddings
uv run cli train --save-embeddings
# Save relation weights
uv run cli train --save-relation-weights
