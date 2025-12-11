#!/bin/bash

# Function to set up environment for O2
setup_o2() {
  local choice=$1

  module load gcc/9.2.0 cuda/11.7 python/3.9.14 git/2.9.5
  conda deactivate
  source proton_env/bin/activate

  if [ "$choice" == "1" ]; then
    echo "Running without Jupyter server on O2..."

  elif [ "$choice" == "2" ]; then
    echo "Running with Jupyter server on O2..."
    jupyter server --no-browser # notebook --port=54321 --browser='none'

  elif [ "$choice" == "3" ]; then
    echo "Running with Seurat on O2..."
    module load R/4.2.1 geos/3.10.2

  else
    echo "Invalid choice for O2."
    exit 1
  fi
}

# Function to set up environment for Kempner
setup_kempner() {
  local choice=$1
  module load gcc/13.2.0-fasrc01 cuda/11.8.0-fasrc01 python/3.10.13-fasrc01 R/4.3.3-fasrc01
  nvcc --version

  if [ "$choice" == "1" ]; then
    echo "Running without Jupyter server on Kempner..."
    conda deactivate
    source /n/home13/anoori/venvs/proton_env/bin/activate

  elif [ "$choice" == "2" ]; then
    echo "Running with Jupyter server on Kempner..."
    conda deactivate
    source /n/home13/anoori/venvs/proton_env/bin/activate
    jupyter server --no-browser # notebook --port=54321 --browser='none'

  elif [ "$choice" == "3" ]; then
    echo "Saving environment on Kempner..."
    conda deactivate
    source /n/home13/anoori/venvs/proton_env/bin/activate
    pip freeze >requirements.txt

  elif [ "$choice" == "4" ]; then
    echo "Setting up virtual environment for PaperQA..."
    module load python/3.12.5-fasrc01
    echo "Python 3.12.5 loaded."
    conda deactivate
    source /n/home13/anoori/venvs/pqa_env/bin/activate
    echo "PaperQA environment activated."

  elif [ "$choice" == "new_pqa" ]; then

    echo "Setting up new environment for PaperQA on Kempner..."
    module load python/3.12.5-fasrc01
    conda deactivate
    cd /n/home13/anoori/venvs
    python -m venv pqa_env
    source pqa_env/bin/activate
    pip install pqapi numpy pandas python-dotenv tqdm
    cd /n/holylfs06/LABS/mzitnik_lab/Users/anoori/PROTON
    echo "PaperQA environment created at: /n/home13/anoori/venvs/pqa_env"

  elif [ "$choice" == "new" ]; then

    echo "Setting up new environment on Kempner..."
    cd /n/home13/anoori/venvs
    virtualenv proton_env
    source proton_env/bin/activate

    pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    pip install torch-geometric
    pip install jupyter jupyterlab notebook
    pip install pandas scipy matplotlib seaborn
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
    python -m pip install lightning
    pip install wandb
    pip install pynvml

    pip install pydantic
    pip install transformers
    pip install scikit-learn
    pip install tqdm
    pip install rapidfuzz

  else
    echo "Invalid choice for Kempner."
    exit 1
  fi
}

# Ask the user if they are on O2 or Kempner
echo "Are you on O2 or Kempner?"
echo "1) O2"
echo "2) Kempner"
read -p "Enter your choice (1 or 2): " env_choice

# Based on the environment, ask the corresponding follow-up questions and run the setup
if [ "$env_choice" == "1" ]; then
  echo "You selected O2."
  echo "What do you want to do?"
  echo "1) Set up virtual environment"
  echo "2) Set up virtual environment and launch Jupyter server"
  echo "3) Set up virtual environment for Seurat"
  read -p "Enter your choice (1, 2, or 3): " o2_choice
  setup_o2 $o2_choice

elif [ "$env_choice" == "2" ]; then
  echo "You selected Kempner."
  echo "What do you want to do?"
  echo "1) Set up virtual environment"
  echo "2) Set up virtual environment and launch Jupyter server"
  echo "3) Take snapshot of virtual environment"
  echo "4) Set up virtual environment for PaperQA"
  read -p "Enter your choice (1, 2, 3, or 4): " kempner_choice
  setup_kempner $kempner_choice

else
  echo "Invalid environment choice."
  exit 1
fi
