import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias, cast

import igraph as ig
import polars as pl

from src.config import conf

_logger = logging.getLogger(__name__)

NodeId: TypeAlias = int
NodesDataFrame: TypeAlias = pl.DataFrame
EdgesDataFrame: TypeAlias = pl.DataFrame


def load_neurokg_data_frames(nodes_file: Path, edges_file: Path) -> tuple[NodesDataFrame, EdgesDataFrame]:
    if not nodes_file.exists():
        raise ValueError(f"Nodes file does not exist at provided path '{nodes_file}'")

    if not edges_file.exists():
        raise ValueError(f"Edges files does not exist at provided path '{edges_file}'")

    _logger.debug(f"Loading nodes from {nodes_file}...")

    NODE_INDEX_POLARS_TYPE = pl.Int64()
    NODE_ID_POLARS_TYPE = pl.String()
    NODE_TYPE_POLARS_TYPE = pl.Enum([
        "effect/phenotype",
        "biological_process",
        "anatomy",
        "exposure",
        "cell_cluster",
        "disease",
        "pathway",
        "cell_type",
        "brain_region",
        "cell_subtype",
        "drug",
        "gene/protein",
        "cellular_component",
        "cell_subcluster",
        "molecular_function",
        "brain_structure",
    ])
    NODE_NAME_POLARS_TYPE = pl.String()
    NODE_SOURCE_POLARS_TYPE = pl.Enum([
        "DrugBank",
        "NCBI",
        "Siletti et al.",
        "HPO",
        "GO",
        "CTD",
        "Kamath et al.",
        "UBERON",
        "MONDO",
        "REACTOME",
    ])

    nodes_df = pl.read_csv(
        nodes_file,
        columns=["node_index", "node_id", "node_type", "node_name", "node_source"],
        new_columns=["id", "external_id", "type", "name", "source"],
        schema=pl.Schema({
            "node_index": NODE_INDEX_POLARS_TYPE,
            "node_id": NODE_ID_POLARS_TYPE,
            "node_type": NODE_TYPE_POLARS_TYPE,
            "node_name": pl.String(),
            "node_source": NODE_SOURCE_POLARS_TYPE,
        }),
    )

    _logger.debug(f"Loading edges from {edges_file}...")

    EDGE_INDEX_POLARS_TYPE = pl.Int64()
    EDGE_ID_POLARS_TYPE = pl.String()
    EDGE_DIRECTION_POLARS_TYPE = pl.Enum(["forward", "reverse"])
    EDGE_RELATION_POLARS_TYPE = pl.String()  # NOTE: Enum not neeced because column is dropped immediatelly
    EDGE_DISPLAY_RELATION_POLARS_TYPE = pl.Enum([
        "transporter",
        "linked to",
        "expression present",
        "phenotype absent",
        "expression absent",
        "off-label use",
        "parent-child",
        "cell type present",
        "indication",
        "upregulated in PD",
        "clinical candidate",
        "ppi",
        "enzyme",
        "side effect",
        "contraindication",
        "interacts with",
        "synergistic interaction",
        "carrier",
        "phenotype present",
        "expression downregulated",
        "expression upregulated",
        "target",
        "associated with",
        "downregulated in PD",
    ])
    EDGE_FULL_RELATION_POLARS_TYPE = pl.String()  # NOTE: Enum not neeced because column is dropped immediatelly

    edges_df = pl.read_csv(
        edges_file,
        columns=["edge_index", "direction", "display_relation", "x_index", "y_index"],
        new_columns=["id", "direction", "edge_type", "tail_node_id", "head_node_id"],
        schema=pl.Schema({
            "edge_index": EDGE_INDEX_POLARS_TYPE,
            "index_label": EDGE_ID_POLARS_TYPE,
            "direction": EDGE_DIRECTION_POLARS_TYPE,
            "relation": EDGE_RELATION_POLARS_TYPE,
            "display_relation": EDGE_DISPLAY_RELATION_POLARS_TYPE,
            "full_relation": EDGE_FULL_RELATION_POLARS_TYPE,
            "x_index": NODE_INDEX_POLARS_TYPE,
            "x_id": NODE_ID_POLARS_TYPE,
            "x_type": NODE_TYPE_POLARS_TYPE,
            "x_name": NODE_NAME_POLARS_TYPE,
            "x_source": NODE_SOURCE_POLARS_TYPE,
            "y_index": NODE_INDEX_POLARS_TYPE,
            "y_id": NODE_ID_POLARS_TYPE,
            "y_type": NODE_TYPE_POLARS_TYPE,
            "y_name": NODE_NAME_POLARS_TYPE,
            "y_source": NODE_SOURCE_POLARS_TYPE,
        }),
    )

    _logger.debug(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges.")

    return nodes_df, edges_df


def load_graph_from_data_frames(nodes_df: NodesDataFrame, edges_df: EdgesDataFrame):
    # Create empty graph
    graph = ig.Graph(directed=True)

    # Add vertices with their attributes
    _logger.debug("Loading nodes into in-memory graph...")
    graph.add_vertices(len(nodes_df))

    # Add edges using the source and target columns
    edges_list: list[tuple[int, int]] = list(zip(edges_df["tail_node_id"], edges_df["head_node_id"], strict=False))

    # If we have node IDs, we need to map them to vertex indices
    node_id_to_idx = {id_val: idx for idx, id_val in cast(Iterator[tuple[int, NodeId]], enumerate(nodes_df["id"]))}
    edges_list = [(node_id_to_idx[src], node_id_to_idx[tgt]) for src, tgt in edges_list]

    _logger.debug("Loading edges into in-memory graph...")
    graph.add_edges(edges_list)

    _logger.debug("Graph fully constructed 🤗")

    return graph


def perform_random_walks(
    nodes_file: Path,
    edges_file: Path,
    start_node_id: int,
    walk_length: int,
    num_walks: int = 10000,
) -> None:
    nodes_df, edges_df = load_neurokg_data_frames(nodes_file, edges_file)
    graph = load_graph_from_data_frames(nodes_df, edges_df)
    graph.to_undirected(mode="collapse")  # Turn into undirected graph, collapsing mutual directed edges

    counter = Counter[NodeId]()
    for _ in range(num_walks):
        nodes_in_walk = cast(
            list[int],
            graph.random_walk(start_node_id, walk_length, mode="out", stuck="return"),
        )
        counter.update(set(nodes_in_walk))

    node_counts_df = pl.DataFrame(counter.most_common(), schema=["id", "visit_count"])
    nodes_with_count_df = nodes_df.join(node_counts_df, on="id", how="left").with_columns(
        pl.col("visit_count").fill_null(0)
    )

    _logger.debug(nodes_with_count_df)

    # Convert Enum columns to String to avoid unsigned dictionary index issues with pandas
    nodes_with_count_df = nodes_with_count_df.with_columns([
        pl.col("type").cast(pl.String()),
        pl.col("source").cast(pl.String()),
    ])

    output_path = Path(conf.paths.notebooks.asyn_screens_dir) / "PD_random_walk_node_visit_count.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_with_count_df.write_parquet(output_path)
