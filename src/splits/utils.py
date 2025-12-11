import logging
import re
import shutil

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import conf
from src.constants import TORCH_DEVICE

_logger = logging.getLogger(__name__)


def _is_roman(s: str) -> bool:
    """Checks if a string is a valid Roman numeral.

    Args:
        s: Input string to check.

    Returns:
        True if the string is a valid Roman numeral, False otherwise.
    """
    return bool(re.search(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", s))


def _clean_disease_names(name: str) -> str:
    """Clean and standardize disease names by removing common suffixes and formatting.

    Args:
        name: Raw disease name string to clean.

    Returns:
        Cleaned disease name with suffixes removed and standardized formatting.

    Examples:
        >>> _clean_disease_names("Alzheimer disease type III")
        'Alzheimer disease'
        >>> _clean_disease_names("Parkinson's disease (disease)")
        "Parkinson's disease"
    """
    name_split = name.split(" ")
    end = name_split[-1]
    if len(end) <= 2 or end.isnumeric() or _is_roman(end):
        name = " ".join(name_split[:-1])

    if name.endswith("type"):
        name = name[:-4]
    if name.endswith(" "):
        name = name[:-1]
    if name.endswith(","):
        name = name[:-1]

    name = name.replace(" (disease)", "")
    name = name.replace("  ", " ")

    return name


def _get_disease_embeddings(
    disease_names: list[str],
) -> np.ndarray:
    """Get disease embeddings using Clinical BioBERT, loading from cache if available.

    Args:
        disease_names: List of disease names to embed.

    Returns:
        Numpy array of disease embeddings.
    """
    conf.paths.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)

    if conf.paths.embeddings_cache_path.is_file():
        _logger.debug(f"Loading cached embeddings from {conf.paths.embeddings_cache_path}...")
        return np.load(conf.paths.embeddings_cache_path)

    _logger.debug(f"Generating embeddings for {len(disease_names)} diseases...")
    tokenizer = AutoTokenizer.from_pretrained(conf.splits.model_name)
    model = AutoModel.from_pretrained(conf.splits.model_name).to(TORCH_DEVICE)
    model.eval()

    all_embeds = []
    batch_size = conf.splits.batch_size
    for i in tqdm(
        range(0, len(disease_names), batch_size),
        desc="Generating embeddings",
        total=(len(disease_names) + batch_size - 1) // batch_size,
    ):
        batch_names = disease_names[i : i + batch_size]
        inputs = tokenizer(batch_names, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            TORCH_DEVICE
        )

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            batch_embeds = outputs.last_hidden_state.mean(dim=1)
            all_embeds.append(batch_embeds.cpu())

    embeddings = torch.cat(all_embeds, dim=0).numpy()
    np.save(conf.paths.embeddings_cache_path, embeddings)
    return embeddings


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes and edges from CSV files.

    Returns:
        Tuple containing:
            nodes: DataFrame of nodes with node_index, node_name, and node_type columns.
            edges: DataFrame of edges with edge_index, x_index, y_index, and edge_type columns.
    """
    nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    edges = pd.read_csv(
        conf.paths.kg.edges_path,
        dtype={"edge_index": int, "x_index": int, "y_index": int},
        low_memory=False,
    )
    return nodes, edges


def preprocess_diseases(
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter for disease nodes, clean names, and get embeddings.

    Args:
        nodes: DataFrame of nodes with node_index, node_name, and node_type columns.

    Returns:
        Tuple containing:
            disease_nodes: DataFrame of disease nodes with clean_name and embedding_score columns.
            embeds: Numpy array of disease embeddings.
    """
    disease_nodes = nodes[nodes["node_type"] == "disease"].reset_index(drop=True)
    input_text = [_clean_disease_names(name) for name in disease_nodes["node_name"].tolist()]
    disease_nodes["clean_name"] = input_text
    embeds = _get_disease_embeddings(input_text)
    return disease_nodes, embeds


def create_disease_groups(
    disease_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    embeds: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Calculate similarity scores and create disease groups.

    Args:
        disease_nodes: DataFrame of disease nodes with node_index, node_name, and node_type columns.
        edges: DataFrame of edges with edge_index, x_index, y_index, and edge_type columns.
        embeds: Numpy array of disease embeddings.

    Returns:
        Dictionary mapping disease names to their corresponding split dataframes.
    """
    cos_sim = cosine_similarity(embeds, embeds)
    disease_split_indices = list(conf.splits.disease_split.values())
    test_set_diseases = disease_nodes[disease_nodes["node_index"].isin(disease_split_indices)].reset_index(drop=True)

    disease_groups = {}
    disease_edges = edges[edges["x_type"] == "disease"]

    _logger.debug(f"Creating disease splits for {len(test_set_diseases)} diseases...")
    for _, row in tqdm(
        test_set_diseases.iterrows(),
        total=len(test_set_diseases),
        desc="Disease splits",
    ):
        disease_group = disease_nodes.copy()

        # Add embedding similarity
        disease_idx = disease_nodes[disease_nodes["node_index"] == row["node_index"]].index[0]
        disease_sim = cos_sim[disease_idx]
        disease_group["embedding_score"] = disease_sim

        # Add Levenshtein distance
        disease_group["levenshtein_score"] = [
            fuzz.token_set_ratio(row["clean_name"], name) for name in disease_group["clean_name"]
        ]

        # Add neighborhood similarity
        neighborhood = disease_edges[disease_edges["x_index"] == row["node_index"]]["y_index"].unique()
        neighborhood_sim = []
        for disease_idx_loop in tqdm(
            disease_nodes["node_index"],
            total=len(disease_nodes),
            desc="Neighborhood similarity",
            leave=False,
        ):
            neighborhood_i = disease_edges[disease_edges["x_index"] == disease_idx_loop]["y_index"].unique()
            union_len = len(set(neighborhood) | set(neighborhood_i))
            jaccard_i = len(set(neighborhood) & set(neighborhood_i)) / union_len if union_len > 0 else 0.0
            neighborhood_sim.append(jaccard_i)
        disease_group["neighborhood_score"] = neighborhood_sim

        def get_methods(r):
            methods = []
            if r["embedding_score"] > conf.splits.embedding_threshold:
                methods.append("embedding")
            if r["levenshtein_score"] > conf.splits.levenshtein_threshold:
                methods.append("levenshtein")
            if r["neighborhood_score"] > conf.splits.neighborhood_threshold:
                methods.append("neighborhood")
            return ", ".join(methods) if methods else np.nan

        disease_group["method"] = disease_group.apply(get_methods, axis=1)

        # Get disease flagged by at least one method
        disease_group = disease_group[disease_group["method"].notna()]

        # Sort by neighborhood score, then by embedding score, then by Levenshtein score
        disease_group = disease_group.sort_values(
            ["neighborhood_score", "embedding_score", "levenshtein_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        # Bring row with disease to the front
        is_target_disease = disease_group["node_index"] == row["node_index"]
        disease_group["is_target_disease"] = is_target_disease
        disease_group = (
            disease_group.sort_values(by=["is_target_disease"], ascending=False)
            .drop(columns=["is_target_disease"])
            .reset_index(drop=True)
        )

        # Add new column
        disease_group["disease_split"] = row["clean_name"]
        disease_group["disease_split_index"] = row["node_index"]
        disease_groups[row["node_name"]] = disease_group

    return disease_groups


def save_splits(
    disease_splits: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    """Save disease splits and return edge counts and all split edges.

    Args:
        disease_splits: DataFrame of disease splits with disease_split_index, node_index, and disease_split columns.
        edges: DataFrame of edges with x_index, y_index, and edge_type columns.

    Returns:
        Tuple containing:
            disease_splits: DataFrame of disease splits with disease_split_index, node_index, and disease_split columns.
            edge_count: Dictionary mapping disease split indices to their corresponding edge counts.
            all_split_edges: DataFrame of all split edges with x_index, y_index, and edge_type columns.
    """
    split_dir = conf.paths.splits_dir / "edges"
    if split_dir.is_dir():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    drug_disease_edges = edges[(edges["x_type"] == "disease") & (edges["y_type"] == "drug")]

    disease_splits_grouped = disease_splits.groupby("disease_split_index")
    edge_count = {}

    for disease_split_index, disease_split_df in tqdm(disease_splits_grouped, desc="Save splits"):
        disease_split_edges = drug_disease_edges[drug_disease_edges["x_index"].isin(disease_split_df["node_index"])]

        if len(disease_split_edges) > 0:
            disease_split_edges.to_csv(
                split_dir / f"{disease_split_index}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            edge_count[disease_split_index] = len(disease_split_edges)

        else:
            disease_splits = disease_splits[~disease_splits["disease_split_index"].isin(disease_split_df["node_index"])]

            tqdm.write(
                f"Removed split {disease_split_index} ({disease_split_df['disease_split'].values[0]}) as it has no edges."
            )

    disease_splits.to_csv(
        conf.paths.splits_dir / "disease_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_split_edges = drug_disease_edges[drug_disease_edges["x_index"].isin(disease_splits["node_index"].unique())]
    all_split_edges.to_csv(split_dir / "all.csv", index=False, encoding="utf-8-sig")

    return disease_splits, edge_count, all_split_edges


def create_and_save_summary(
    disease_splits: pd.DataFrame,
    all_split_edges: pd.DataFrame,
    edge_count: dict[int, int],
):
    """Create and save a summary of the disease splits.

    Args:
        disease_splits: DataFrame of disease splits with disease_split_index, node_index, and disease_split columns.
        all_split_edges: DataFrame of all split edges with x_index, y_index, and edge_type columns.
        edge_count: Dictionary mapping disease split indices to their corresponding edge counts.
    """
    splits_df = disease_splits[disease_splits["node_index"] == disease_splits["disease_split_index"]]
    splits_df = splits_df.drop_duplicates(subset="disease_split_index").reset_index(drop=True)
    splits_df = splits_df[["node_index", "node_id", "node_type", "node_name", "disease_split"]]

    disease_splits_grouped = (
        disease_splits.groupby("disease_split_index").size().reset_index(name="node_count")  # ty: ignore[no-matching-overload]
    )
    splits_df = splits_df.merge(
        disease_splits_grouped,
        left_on="node_index",
        right_on="disease_split_index",
        how="left",
    )
    splits_df = splits_df.drop(columns="disease_split_index")

    splits_df["edge_count"] = splits_df["node_index"].map(edge_count)

    # Add row for all to beginning
    total_nodes = len(disease_splits["node_index"].unique())
    total_edges = len(all_split_edges)
    splits_df = pd.concat(
        [
            pd.DataFrame({
                "node_index": ["all"],
                "node_id": [None],
                "node_type": ["disease"],
                "node_name": ["all"],
                "disease_split": ["all"],
                "node_count": [total_nodes],
                "edge_count": [total_edges],
            }),
            splits_df,
        ],
        axis=0,
    ).reset_index(drop=True)

    splits_summary_path = conf.paths.splits_dir / "disease_splits_summary.csv"
    _logger.debug(f"Saving disease splits summary to {splits_summary_path}...")
    splits_df.to_csv(
        splits_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
