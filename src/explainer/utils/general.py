import logging

import dgl
import torch
import wandb

from src.config import conf
from src.constants import TORCH_DEVICE
from src.models import HGT

_logger = logging.getLogger(__name__)


def get_node_name(nodes, node_index):
    return nodes.loc[nodes["node_index"] == node_index, "node_name"].values[0]


def load_model_from_checkpoint(kg):
    checkpoint_path = conf.paths.checkpoint.checkpoint_path
    _logger.debug(f"Loading model from checkpoint {checkpoint_path}")
    model = HGT.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        kg=kg,
        hparams=conf.neurokg,
        strict=False,
    )
    model.eval()
    model = model.to(TORCH_DEVICE)
    return model


def get_original_score(
    model: HGT,
    sg: dgl.DGLGraph,
    query_edge_graph,
    query_edge_type: tuple[str, str, str],
):
    with torch.no_grad():
        node_embeddings = model.forward(sg)
        scores = model.decoder(sg, query_edge_graph, node_embeddings)
        original_score = torch.sigmoid(scores[query_edge_type])
        return original_score


def init_wandb(run_name, save_path, explainer_hparams, src_name, dst_name, query_edge_type, sg):
    wandb.init(
        project="gnn-explainer",
        name=run_name,
        dir=save_path,
        config={
            "lr": explainer_hparams.lr,
            "num_epochs": explainer_hparams.num_epochs,
            "sparsity_loss_alpha": explainer_hparams.sparsity_loss_alpha,
            "entropy_loss_alpha": explainer_hparams.entropy_loss_alpha,
            "khop": explainer_hparams.khop,
            "degree_threshold": explainer_hparams.degree_threshold,
            "src_node": src_name,
            "dst_node": dst_name,
            "query_edge_type": query_edge_type,
            "num_nodes": sg.num_nodes(),
            "num_edges": sg.num_edges(),
        },
    )
