import logging
from datetime import datetime

import dgl
import pandas as pd
import pytorch_lightning as pl
import torch

from src.config import conf
from src.dataloaders import load_graph
from src.models import HGT

from .pretrain import pretrain

_logger = logging.getLogger(__name__)


@torch.no_grad()
def save_embeddings() -> None:
    """Computes and saves the node embeddings from a trained HGT model."""
    _logger.info("Saving embeddings...")

    pl.seed_everything(conf.seed, workers=True)

    kg: dgl.DGLHeteroGraph = load_graph()

    checkpoint_path = conf.paths.checkpoint.checkpoint_path
    _logger.info(f"Loading model from checkpoint {checkpoint_path}")
    model: HGT = HGT.load_from_checkpoint(  # type: ignore[call-arg]
        checkpoint_path=str(checkpoint_path),
        kg=kg,
        hparams=conf.neurokg,
        strict=False,
    )
    model.eval()

    model.cache_graph(kg, overwrite=False, degree_threshold=conf.explainer.degree_threshold)

    embed_dir = conf.paths.checkpoint.embeddings_path.parent
    _logger.info(f"Saving embeddings to {embed_dir}")
    embed_dir.mkdir(parents=True, exist_ok=True)

    node_ids: list[int] = list(range(model.emb.weight.shape[0]))
    chunk_size = 1000
    all_embeddings: list[torch.Tensor] = []

    for i in range(0, len(node_ids), chunk_size):
        end_index = min(i + chunk_size, len(node_ids))
        progress_percent = round((i / len(node_ids)) * 100, 2)
        current_time = datetime.now().strftime("%H:%M:%S on %m/%d/%Y")
        _logger.info(
            f"Processing nodes {i} to {end_index - 1} out of {len(node_ids) - 1} "
            f"({progress_percent}%) at {current_time}"
        )

        chunk_path = embed_dir / f"embeddings_{i}_{end_index - 1}.pt"
        if chunk_path.exists():
            _logger.info(f"Skipping chunk {i} to {end_index - 1} as it already exists.")
            chunk_embeddings = torch.load(chunk_path)
        else:
            chunk_node_ids = node_ids[i:end_index]
            chunk_embeddings = model.get_embeddings(query_indices=chunk_node_ids)
            torch.save(chunk_embeddings, chunk_path)

        all_embeddings.append(chunk_embeddings)

    embeddings = torch.cat(all_embeddings, dim=0)

    # Save as .pt file
    embed_path_pt = conf.paths.checkpoint.embeddings_path
    torch.save(embeddings, embed_path_pt)
    _logger.info(f"Saved embeddings with shape: {embeddings.shape} to {embed_path_pt}")

    # Save as .csv file
    _logger.info("Saving embeddings to CSV...")
    embeddings_df = pd.DataFrame(embeddings.cpu().detach().numpy())
    embed_path_csv = embed_dir / "embeddings.csv"
    embeddings_df.to_csv(embed_path_csv, index=False)
    _logger.info(f"Saved embeddings to {embed_path_csv}")


@torch.no_grad()
def save_embedding_layer() -> None:
    """Saves the first embedding layer of a trained HGT model."""
    _logger.info("Saving first embedding layer...")

    pl.seed_everything(conf.seed, workers=True)

    kg = load_graph()

    checkpoint_path = conf.paths.checkpoint.checkpoint_path
    _logger.info(f"Loading model from checkpoint {checkpoint_path}")
    model = HGT.load_from_checkpoint(  # type: ignore[call-arg]
        checkpoint_path=str(checkpoint_path),
        kg=kg,
        hparams=conf.neurokg,
        strict=False,
    )
    model.eval()

    n_id = torch.arange(model.emb.weight.shape[0])
    embeddings = model.emb(n_id)

    embed_dir = conf.paths.checkpoint.base_dir / "embeddings" / checkpoint_path.stem
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Save embedding layer
    embed_path_pt = embed_dir / "embedding_layer.pt"
    torch.save(embeddings, embed_path_pt)
    _logger.info(f"Saved embedding layer with shape: {embeddings.shape} to {embed_path_pt}")

    _logger.info("Saving embedding layer to CSV...")
    embeddings_df = pd.DataFrame(embeddings.cpu().detach().numpy())
    embed_path_csv = embed_dir / "embedding_layer.csv"
    embeddings_df.to_csv(embed_path_csv, index=False)
    _logger.info(f"Saved embedding layer to {embed_path_csv}")


@torch.no_grad()
def save_decoder() -> None:
    """Saves the decoder of a trained HGT model."""
    _logger.info("Saving decoder...")

    pl.seed_everything(conf.seed, workers=True)

    kg = load_graph()

    checkpoint_path = conf.paths.checkpoint.checkpoint_path
    _logger.info(f"Loading model from checkpoint {checkpoint_path}")
    model = HGT.load_from_checkpoint(  # type: ignore[call-arg]
        checkpoint_path=str(checkpoint_path),
        kg=kg,
        hparams=conf.neurokg,
        strict=False,
    )
    model.eval()

    decoder = model.decoder.relation_weights.cpu().detach().numpy()
    edge_types = kg.canonical_etypes

    embed_dir = conf.paths.checkpoint.embeddings_path.parent
    embed_dir.mkdir(parents=True, exist_ok=True)

    weights_path = embed_dir / "decoder.pt"
    edge_types_path = embed_dir / "edge_types.pt"
    torch.save(decoder, weights_path)
    _logger.info(f"Saved decoder with shape: {decoder.shape} to {weights_path}")
    torch.save(edge_types, edge_types_path)
    _logger.info(f"Saved edge types with length: {len(edge_types)} to {edge_types_path}")


def run_train():
    if conf.proton.training.output_options.save_embeddings:
        save_embeddings()
    elif conf.proton.training.output_options.save_embedding_layer:
        save_embedding_layer()
    elif conf.proton.training.output_options.save_decoder:
        save_decoder()
    else:
        pretrain()
