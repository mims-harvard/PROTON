import logging

import torch

from src.config import conf
from src.constants import TORCH_DEVICE

logger = logging.getLogger(__name__)


def get_cosine_similarity() -> None:
    embeddings_path = conf.paths.checkpoint.embeddings_path
    logger.debug(f"Loading embeddings from {embeddings_path}...")
    embeddings = torch.load(embeddings_path)

    logger.debug(f"Moving embeddings to {TORCH_DEVICE}...")
    embeddings = embeddings.to(TORCH_DEVICE)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    logger.debug("Computing cosine similarity...")
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    sim_path = embeddings_path.parent / f"{embeddings_path.stem}_cos_sim.pt"
    torch.save(sim_matrix, sim_path)

    logger.debug(f"Saved similarity matrix to {sim_path}")
