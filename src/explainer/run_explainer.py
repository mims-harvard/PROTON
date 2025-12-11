import logging
import os
import types
import uuid

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from src.config import conf
from src.constants import TORCH_DEVICE
from src.dataloaders import load_graph

from .gnn_explainer import GNNExplainer
from .utils.general import (
    get_node_name,
    get_original_score,
    init_wandb,
    load_model_from_checkpoint,
)
from .utils.graph import model_khop_in_subgraph
from .utils.patching import forward_exp, model_forward_exp

_logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    dgl.seed(seed)
    pl.seed_everything(seed, workers=True)


def run_explainer():
    _set_seed(conf.seed)
    nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    edges = pd.read_csv(
        conf.paths.kg.edges_path,
        dtype={"edge_index": int, "x_index": int, "y_index": int},
        low_memory=False,
    )
    kg = load_graph(nodes, edges)
    _logger.info(f"Number of nodes: {kg.num_nodes()}")
    _logger.info(f"Number of edges: {kg.num_edges()}")

    model = load_model_from_checkpoint(kg)

    model.conv1.forward = types.MethodType(forward_exp, model.conv1)
    model.conv2.forward = types.MethodType(forward_exp, model.conv2)
    model.conv3.forward = types.MethodType(forward_exp, model.conv3)
    model.forward = types.MethodType(model_forward_exp, model)

    src_index = conf.explainer.src_indices[0]
    dst_index = conf.explainer.dst_indices[0]
    query_edge_type = conf.explainer.query_edge_types[0]

    src_name = get_node_name(nodes, src_index)
    dst_name = get_node_name(nodes, dst_index)
    _logger.info(f"Explaining edge {src_name} → {dst_name}")

    khop_sg, query_edge_graph, _ = model_khop_in_subgraph(
        kg, conf.explainer.khop, [src_index], [dst_index], query_edge_type
    )

    degree_mask = khop_sg.out_degrees() < conf.explainer.degree_threshold
    degree_mask = degree_mask | (khop_sg.ndata["node_index"] == src_index) | (khop_sg.ndata["node_index"] == dst_index)
    sg = dgl.node_subgraph(khop_sg, degree_mask, relabel_nodes=False)

    _logger.info(f"Number of nodes in subgraph: {sg.num_nodes()}")
    _logger.info(f"Number of edges in subgraph: {sg.num_edges()}")

    original_score = (
        get_original_score(
            model,
            sg.to(TORCH_DEVICE),
            query_edge_graph.to(TORCH_DEVICE),
            query_edge_type,
        )
        .cpu()
        .numpy()[0]
    )

    _logger.info(f"Original score: {original_score}")

    run_uuid = str(uuid.uuid4())[:4]

    run_name = (
        f"{src_name} → {dst_name} | "
        f"{conf.explainer.khop} khop. | "
        f"{conf.explainer.lr} lr. | "
        f"{conf.explainer.sparsity_loss_alpha} s. | "
        f"{conf.explainer.entropy_loss_alpha} ent. | "
        f"{conf.explainer.num_epochs} epochs | "
        f"{run_uuid}"
    )
    save_path = conf.paths.explainer_dir / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    init_wandb(
        run_name,
        save_path,
        conf.explainer,
        src_name,
        dst_name,
        query_edge_type,
        sg,
    )

    for relation in ["positive"]:
        explainer = GNNExplainer(
            model=model,
            original_score=original_score,
            sg=sg,
            query_edge_graph=query_edge_graph,
            query_edge_type=query_edge_type,
            src_name=src_name,
            dst_name=dst_name,
            explainer_hparams=conf.explainer,
            kg=kg,
            nodes=nodes,
            edges=edges,
            save_path=save_path,
            relation=relation,
        )
        explainer.explain()

    wandb.finish()
