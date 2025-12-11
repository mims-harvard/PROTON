from enum import Enum
from pathlib import Path

import torch

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONF_DIR: Path = REPO_ROOT / "conf"
DEFAULT_CONFIG_PATH: Path = CONF_DIR / "default.config.yaml"
DATA_DIR: Path = REPO_ROOT / "data"

if torch.cuda.is_available():
    TORCH_DEVICE: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    TORCH_DEVICE: torch.device = torch.device("mps")
else:
    TORCH_DEVICE: torch.device = torch.device("cpu")

DRUGKG_DIR: Path = DATA_DIR / "drugkg"
NEUROKG_DIR: Path = DATA_DIR / "neurokg"
FINETUNING_DIR: Path = DATA_DIR / "finetuning"


class KG(str, Enum):
    DRUGKG = "DrugKG"
    NEUROKG = "NeuroKG"
