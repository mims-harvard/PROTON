import logging

import dgl
import numpy as np
import pandas as pd
import torch

from src.config import conf
from src.samplers import FixedSampler

logger = logging.getLogger(__name__)


def load_graph(nodes: pd.DataFrame | None = None, edges: pd.DataFrame | None = None) -> dgl.DGLHeteroGraph:
    """Load a graph from nodes and edges.

    Args:
        nodes: DataFrame of nodes with node_index, node_name, and node_type columns.
        edges: DataFrame of edges with edge_index, x_index, y_index, and edge_type columns.

    Returns:
        DGLHeteroGraph object representing the graph.
    """
    if nodes is None:
        nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    if edges is None:
        edges = pd.read_csv(
            conf.paths.kg.edges_path, dtype={"edge_index": int, "x_index": int, "y_index": int}, low_memory=False
        )

    nodes["node_type_index"] = nodes.groupby("node_type").cumcount()

    edges["x_type_index"] = nodes.loc[edges["x_index"], "node_type_index"].values
    edges["y_type_index"] = nodes.loc[edges["y_index"], "node_type_index"].values

    kg_data = {
        (x_type, relation, y_type): (
            torch.tensor(group["x_type_index"].values),
            torch.tensor(group["y_type_index"].values),
        )
        for (x_type, relation, y_type), group in edges.groupby(["x_type", "relation", "y_type"])  # ty: ignore[not-iterable]
    }

    kg = dgl.heterograph(kg_data)

    for node_type, group in nodes.groupby("node_type"):
        kg.nodes[node_type].data["node_index"] = torch.tensor(group["node_index"].values)

    # Optionally subsample high-degree nodes
    if conf.proton.training.graph_sampling.subsample_graph:
        if conf.proton.training.graph_sampling.degree_threshold is None:
            raise ValueError("degree_threshold required when subsample_graph=True")

        # Sum degrees across edge types
        degrees = {}
        for (src_type, _, _), degree in {etype: kg.out_degrees(etype=etype) for etype in kg.canonical_etypes}.items():
            degrees[src_type] = degrees.get(src_type, 0) + degree

        # Filter nodes by degree threshold
        degree_mask = {
            ntype: degree <= conf.proton.training.graph_sampling.degree_threshold for ntype, degree in degrees.items()
        }
        kg = dgl.node_subgraph(kg, degree_mask, relabel_nodes=False)

    return kg


def _load_test_set_eids(kg: dgl.DGLHeteroGraph) -> dict[tuple[str, str, str], torch.Tensor]:
    """Loads test set edge IDs from a file.

    Args:
        kg: DGL heterograph containing the full knowledge graph

    Returns:
        Dictionary mapping edge types (source_type, relation, target_type) to tensors
        of edge IDs that belong to the test set. Returns empty dict if no test set
        path is specified in config.

    Notes:
        - Expects a CSV file with columns: edge_index, x_index, y_index, x_type,
          relation, y_type
        - Maps node indices from the CSV file to internal DGL node IDs
        - Groups edges by type and creates edge ID tensors for each type
    """
    test_set_eids = {}
    if not conf.neurokg.test_set:
        return test_set_eids

    test_set_path = conf.paths.splits_dir / "split_edges_GPT" / (conf.neurokg.test_set + ".csv")
    test_edges = pd.read_csv(
        test_set_path,
        dtype={"edge_index": int, "x_index": int, "y_index": int},
        low_memory=False,
    )
    kg_indices = kg.ndata["node_index"]

    for (x_type, relation, y_type), split_subset in test_edges.groupby(  # ty: ignore[not-iterable]
        ["x_type", "relation", "y_type"], sort=False
    ):
        x_map = dict(zip(kg_indices[x_type].numpy(), kg.nodes(x_type).numpy(), strict=False))
        y_map = dict(zip(kg_indices[y_type].numpy(), kg.nodes(y_type).numpy(), strict=False))

        x_nodes = torch.tensor([x_map[x] for x in split_subset["x_index"]])
        y_nodes = torch.tensor([y_map[y] for y in split_subset["y_index"]])

        edge_ids = kg.edge_ids(x_nodes, y_nodes, etype=(x_type, relation, y_type))
        test_set_eids[(x_type, relation, y_type)] = edge_ids

    return test_set_eids


def _split_edges(
    eids: torch.Tensor,
    test_eids: torch.Tensor | None,
    use_preset_test_set: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits edge IDs into train, validation, and test sets.

    Args:
        eids: Tensor of edge IDs to split.
        test_eids: Optional tensor of predefined test set edge IDs.
        use_preset_test_set: Whether to use predefined test set or create new split.

    Returns:
        Tuple of (train_eids, val_eids, test_eids) tensors containing the split edge IDs.
        Each tensor is sorted in ascending order.
    """
    if use_preset_test_set:
        train_val_eids = eids[~torch.isin(eids, test_eids)] if test_eids is not None else eids
        shuffled_eids = train_val_eids[torch.randperm(train_val_eids.shape[0])]

        # 90/10 split for train/validation
        num_train_val = shuffled_eids.shape[0]
        val_len = int(np.ceil(0.10 * num_train_val))
        train_len = num_train_val - val_len

        train_set = torch.sort(shuffled_eids[:train_len])[0]
        val_set = torch.sort(shuffled_eids[train_len:])[0]
        test_set = torch.sort(test_eids)[0] if test_eids is not None else torch.tensor([])
    else:
        # 80/15/5 split for train/validation/test
        num_edges = eids.shape[0]
        shuffled_eids = eids[torch.randperm(num_edges)]

        test_len = int(np.ceil(0.05 * num_edges))
        val_len = int(np.ceil(0.15 * num_edges))
        train_len = num_edges - test_len - val_len

        train_set = torch.sort(shuffled_eids[:train_len])[0]
        val_set = torch.sort(shuffled_eids[train_len : train_len + val_len])[0]
        test_set = torch.sort(shuffled_eids[train_len + val_len :])[0]

    return train_set, val_set, test_set


def partition_graph(kg: dgl.DGLHeteroGraph) -> tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]:
    """Partitions the graph into training, validation, and test sets.

    Args:
        kg: DGL heterograph containing the full knowledge graph

    Returns:
        Tuple of (train_kg, val_kg, test_kg) DGL heterographs representing the partitioned graph.
    """
    train_eids: dict[tuple[str, str, str], torch.Tensor] = {}
    val_eids: dict[tuple[str, str, str], torch.Tensor] = {}
    test_eids: dict[tuple[str, str, str], torch.Tensor] = {}

    test_set = _load_test_set_eids(kg)
    use_preset_test_set = bool(test_set)

    forward_edge_types = [et for et in kg.canonical_etypes if "rev" not in et[1]]

    for etype in forward_edge_types:
        etype_eids = kg.edges(etype=etype, form="eid")
        etype_test_eids = test_set.get(etype)

        etype_train, etype_val, etype_test = _split_edges(etype_eids, etype_test_eids, use_preset_test_set)

        train_eids[etype] = etype_train
        val_eids[etype] = etype_val
        test_eids[etype] = etype_test

        mask = torch.zeros(etype_eids.shape[0])
        mask[etype_val] = 1
        if etype_test.shape[0] > 0:
            mask[etype_test] = 2

        reverse_etype = (etype[2], f"rev_{etype[1]}", etype[0])
        train_eids[reverse_etype] = etype_train
        val_eids[reverse_etype] = etype_val
        test_eids[reverse_etype] = etype_test

        kg.edges[etype].data["mask"] = mask
        kg.edges[reverse_etype].data["mask"] = mask

    train_kg = kg.edge_subgraph(train_eids, relabel_nodes=False)

    train_val_eids = {etype: torch.sort(torch.cat([train_eids[etype], val_eids[etype]]))[0] for etype in train_eids}
    val_kg = kg.edge_subgraph(train_val_eids, relabel_nodes=False)

    test_kg = kg

    return train_kg, val_kg, test_kg


def create_dataloaders(
    kg: dgl.DGLHeteroGraph,
    train_kg: dgl.DGLHeteroGraph,
    val_kg: dgl.DGLHeteroGraph,
    test_kg: dgl.DGLHeteroGraph,
    sampler_fanout: list[int] = [1, 1, 1],  # noqa: B006
    fixed_k: int = 10,
    negative_k: int = 8,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    test_batch_size: int = 8,
    num_workers: int = 0,
    subsample_graph: bool = False,
    full_kg: bool = False,
) -> tuple[dgl.dataloading.DataLoader | None, dgl.dataloading.DataLoader | None, dgl.dataloading.DataLoader | None]:
    """Create dataloaders for training, validation and testing.

    Args:
        kg: Full knowledge graph
        train_kg: Training subgraph
        val_kg: Validation subgraph
        test_kg: Test subgraph
        sampler_fanout: Number of neighbors to sample per layer
        fixed_k: Number of neighbors to sample for FixedSampler
        negative_k: Number of negative samples per edge
        train/val/test_batch_size: Batch sizes for each dataloader
        num_workers: Number of worker processes
        subsample_graph: Whether to use ShaDowKHopSampler
        full_kg: Whether to use full graph without train/val/test split

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    logger.info("Creating mini-batch dataloaders...")

    # Setup edge type mappings
    forward_edge_types = [x for x in kg.canonical_etypes if "rev" not in x[1]]
    reverse_edge_dict = {(u, r, v): (v, "rev_" + r, u) for u, r, v in forward_edge_types}
    reverse_edge_dict.update({value: key for key, value in reverse_edge_dict.items()})

    if full_kg:
        logger.info("Using full KG for training only")
        sub_eids = {etype: kg.edges(etype=etype, form="eid") for etype in kg.canonical_etypes}
        sampler = FixedSampler(sampler_fanout, fixed_k=fixed_k, upsample_rare_types=True)
        dataloader = _create_dataloader(
            kg, sub_eids, sampler, reverse_edge_dict, negative_k, train_batch_size, num_workers
        )
        return dataloader, None, None

    logger.info("Creating train, validation and test dataloaders")

    # Get edge indices for each split
    sub_eids = {}
    for split_name, split_kg in [("train", train_kg), ("val", val_kg), ("test", test_kg)]:
        split_eids = {}
        for etype in kg.canonical_etypes:
            mask = split_kg.edges[etype].data["mask"] == {"train": 0, "val": 1, "test": 2}[split_name]
            split_eids[etype] = split_kg.edges(etype=etype, form="eid")[mask]
        sub_eids[split_name] = split_eids

    # Create sampler
    if subsample_graph:
        sampler = dgl.dataloading.ShaDowKHopSampler([-1, -1, -1])
    else:
        sampler = FixedSampler(sampler_fanout, fixed_k=fixed_k, upsample_rare_types=True)

    # Create dataloaders
    train_dl = _create_dataloader(
        train_kg, sub_eids["train"], sampler, reverse_edge_dict, negative_k, train_batch_size, num_workers
    )
    val_dl = _create_dataloader(
        val_kg, sub_eids["val"], sampler, reverse_edge_dict, negative_k, val_batch_size, num_workers
    )
    test_dl = _create_dataloader(
        test_kg, sub_eids["test"], sampler, reverse_edge_dict, negative_k, test_batch_size, num_workers
    )

    return train_dl, val_dl, test_dl


def _create_dataloader(
    kg: dgl.DGLHeteroGraph,
    eids: dict[tuple[str, str, str], torch.Tensor],
    sampler: dgl.dataloading.Sampler,
    reverse_edge_dict: dict[tuple[str, str, str], tuple[str, str, str]],
    negative_k: int,
    batch_size: int,
    num_workers: int,
) -> dgl.dataloading.DataLoader:
    """Helper to create a single dataloader with common settings."""
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(negative_k)
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude="reverse_types", reverse_etypes=reverse_edge_dict, negative_sampler=neg_sampler
    )
    return dgl.dataloading.DataLoader(
        kg, eids, edge_sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers
    )
