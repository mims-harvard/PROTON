import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def format_rich(value: str, markup: str) -> str:
    """Format string with rich markup.

    Args:
        value: The string to format.
        markup: The rich markup to apply.

    Returns:
        The formatted string.
    """
    return f"[{markup}]{value}[/{markup}]"


def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float | np.ndarray]:
    """Calculates classification metrics between predictions and targets.

    Args:
        pred: Array of predicted probabilities.
        target: Array of target labels.
        threshold: Threshold for classification.

    Returns:
        Dictionary containing various classification metrics.
    """
    metrics: dict[str, float | np.ndarray] = {
        "accuracy": accuracy_score(target, pred > threshold),
        "ap": average_precision_score(target, pred),
        "f1": f1_score(target, pred > threshold, average="micro"),
        "auroc": roc_auc_score(target, pred) if len(np.unique(target)) > 1 else 0.5,
        "pred": pred,
        "target": target,
    }
    return metrics


def random_walk(
    edges: pd.DataFrame,
    seed_node: int,
    walk_length: int,
    walk_nodes: set[int] | None = None,
) -> set[int]:
    """Performs a random walk on the input graph.

    Args:
        edges: Data frame of edges in graph.
        seed_node: Index of seed node.
        walk_length: Length of random walk.
        walk_nodes: Set of nodes in random walk.

    Returns:
        Set of nodes in the random walk.
    """
    walk_nodes = {seed_node} if walk_nodes is None else walk_nodes | {seed_node}

    if walk_length == 0:
        return walk_nodes

    neighbors = edges[edges.x_index == seed_node].y_index.values
    next_node = np.random.choice(neighbors)

    return random_walk(edges, next_node, walk_length - 1, walk_nodes)


def random_subgraph(edges: pd.DataFrame, seed_node: int, n_walks: int, walk_length: int) -> set[int]:
    """Generates a random subgraph from the input graph.

    Args:
        edges: Data frame of edges in graph.
        seed_node: Index of seed node.
        n_walks: Number of random walks to perform.
        walk_length: Length of each random walk.

    Returns:
        Set of nodes in the random subgraph.
    """
    subgraph_nodes = {seed_node}
    for _ in range(n_walks):
        subgraph_nodes = random_walk(edges, np.random.choice(list(subgraph_nodes)), walk_length, subgraph_nodes)
    return subgraph_nodes


def generate_subgraph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed_node: int,
    n_walks: int,
    walk_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a random subgraph and remaps node indices to be consecutive integers.

    Args:
        nodes: Data frame of nodes in graph.
        edges: Data frame of edges in graph.
        seed_node: Index of seed node.
        n_walks: Number of random walks to perform.
        walk_length: Length of each random walk.

    Returns:
        Tuple of (subgraph_nodes, subgraph_edges) DataFrames.
    """
    subgraph_nodes = nodes[nodes.node_index.isin(random_subgraph(edges, seed_node, n_walks, walk_length))].copy()

    index_map = dict(zip(subgraph_nodes.node_index, range(len(subgraph_nodes)), strict=False))

    subgraph_edges = edges[edges.x_index.isin(index_map) & edges.y_index.isin(index_map)].copy()
    subgraph_edges.x_index = subgraph_edges.x_index.map(index_map)
    subgraph_edges.y_index = subgraph_edges.y_index.map(index_map)

    # Update node indices and reset both DataFrames
    subgraph_nodes.node_index = range(len(subgraph_nodes))
    return subgraph_nodes.reset_index(drop=True), subgraph_edges.reset_index(drop=True)
