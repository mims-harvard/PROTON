#!/bin/bash

# Set permissions with chmod +x sync_results.sh
# Run with ./sync_results.sh

# Ask the user for the environment
echo "Where do you want to sync to?"
echo "1) Kempner --> O2"
echo "2) O2 --> Kempner"
echo "3) local"
read -p "Enter your choice (1, 2, or 3): " env_choice

# Ask the user which results folder to sync
echo "Which results folder do you want to sync?"
echo "1) pretraining"
echo "2) disease_splits"
echo "3) sweep"
read -p "Enter your choice (1, 2, or 3): " folder_choice

# Map user input to folder names
case $folder_choice in
    1) RESULTS_FOLDER="pretraining";;
    2) RESULTS_FOLDER="disease_splits";;
    3) RESULTS_FOLDER="sweep";;
    *) echo "Invalid choice. Please enter 1, 2, or 3."; exit 1;;
esac

# Define directories based on environment
if [ "$env_choice" == "1" ]; then
    # O2 environment
    SRC_DIR="anoori@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON/data/$RESULTS_FOLDER/"
    DEST_DIR="/n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/$RESULTS_FOLDER/"

elif [ "$env_choice" == "2" ]; then
    # Kempner environment
    SRC_DIR="an252@transfer.rc.hms.harvard.edu:/n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/$RESULTS_FOLDER/"
    DEST_DIR="/n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON/data/$RESULTS_FOLDER/"

elif [ "$env_choice" == "3" ]; then
    # Local environment
    LOCAL_DIR="/tmp/sync_results"
    SRC_DIR="anoori@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON/data/$RESULTS_FOLDER/"
    DEST_DIR="an252@transfer.hms.harvard.edu:/n/data1/hms/dbmi/zitnik/lab/users/an252/PROTON/data/$RESULTS_FOLDER/"

    # Create local temporary directory
    mkdir -p $LOCAL_DIR

    # Synchronize from SRC_DIR to LOCAL_DIR
    rsync -avz $SRC_DIR $LOCAL_DIR

    # Synchronize from LOCAL_DIR to DEST_DIR
    rsync -avz $LOCAL_DIR/ $DEST_DIR

    # Remove local temporary directory
    rm -rf $LOCAL_DIR
else
    echo "Invalid choice. Please enter 1, 2, or 3."
    exit 1
fi

# If not local, synchronize directly between remote and local paths
if [ "$env_choice" != "3" ]; then
    # Synchronize from SRC_DIR to DEST_DIR
    rsync -avz -e ssh $SRC_DIR/ $DEST_DIR

    # Synchronize from DEST_DIR to SRC_DIR (to make sure both directories are updated)
    # rsync -avz $DEST_DIR $SRC_DIR
fi

echo "Synchronization complete."
