#!/usr/bin/env bash
#
# transfer_embeddings.sh
# Usage: ./transfer_embeddings.sh <CHECKPOINT_NAME>
# Example: ./transfer_embeddings.sh 2024_07_03_11_19_22_epoch=2-step=60577
#
# This script will:
#   1) Use a single scp command to copy the following files from the remote server:
#       - pretraining/checkpoints/<CHECKPOINT_NAME>.ckpt
#       - pretraining/embeddings/<CHECKPOINT_NAME>_embeddings.pt
#       - pretraining/embeddings/<CHECKPOINT_NAME>_edge_types.pt
#       - pretraining/embeddings/<CHECKPOINT_NAME>_decoder.pt
#   2) Move them locally to the correct folders so they end up in:
#       - local/pretraining/checkpoints/
#       - local/pretraining/embeddings/
#

# -----------------------------
# 1. Parse input & define paths
# -----------------------------
if [ $# -ne 1 ]; then
  echo "Error: Please provide exactly one argument for the checkpoint name."
  echo "Usage: $0 <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1

# Remote server details
REMOTE_USER="an252"
REMOTE_HOST="transfer.rc.hms.harvard.edu"
REMOTE_BASE_PATH="/n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON"

# Local folder where NeuroKG is located
LOCAL_BASE_PATH="/Users/an583/Documents/Zitnik_Lab/PROTON"

# For clarity, define final subfolders we want locally
LOCAL_CHECKPOINTS="${LOCAL_BASE_PATH}/data/checkpoints"
LOCAL_EMBEDDINGS="${LOCAL_BASE_PATH}/data/embeddings"

echo "---------------------------------------"
echo "CHECKPOINT_NAME = ${CHECKPOINT_NAME}"
echo "LOCAL_BASE_PATH = ${LOCAL_BASE_PATH}"
echo "---------------------------------------"

# Make sure local directories exist
mkdir -p "${LOCAL_CHECKPOINTS}"
mkdir -p "${LOCAL_EMBEDDINGS}"

# -----------------------------------------------------
# 2. Transfer the checkpoint file from remote to local
# -----------------------------------------------------
echo "Transferring checkpoint file: ${CHECKPOINT_NAME}.ckpt"
scp -r \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/data/checkpoints/${CHECKPOINT_NAME}.ckpt" \
  "${LOCAL_CHECKPOINTS}/"

# -----------------------------------------------------------------------
# 3. Transfer the three embedding files (embeddings, edge_types, weights)
# -----------------------------------------------------------------------
echo "Transferring embeddings file: ${CHECKPOINT_NAME}_embeddings.pt"
scp -r \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/data/embeddings/${CHECKPOINT_NAME}_embeddings.pt" \
  "${LOCAL_EMBEDDINGS}/"

echo "Transferring edge types file: ${CHECKPOINT_NAME}_edge_types.pt"
scp -r \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/data/embeddings/${CHECKPOINT_NAME}_edge_types.pt" \
  "${LOCAL_EMBEDDINGS}/"

echo "Transferring decoder file: ${CHECKPOINT_NAME}_decoder.pt"
scp -r \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/data/embeddings/${CHECKPOINT_NAME}_decoder.pt" \
  "${LOCAL_EMBEDDINGS}/"
