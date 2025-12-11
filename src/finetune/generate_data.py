import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import conf
from src.constants import DATA_DIR

_logger = logging.getLogger(__name__)


class FineTuneDataset(Dataset):
    def __init__(self, all_pairs: list[tuple[int, int]], all_labels: list[int], embeddings: torch.Tensor) -> None:
        self.all_pairs = all_pairs
        self.all_labels = all_labels
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.all_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _idx1, idx2 = self.all_pairs[idx]
        x = torch.cat([self.embeddings[idx], self.embeddings[idx2]])
        return x, torch.tensor(self.all_labels[idx], dtype=torch.float)


def scramble_pairs_and_labels(
    pairs: list[tuple[int, int]], labels: list[int], seed: int | None = None
) -> tuple[list[tuple[int, int]], list[int]]:
    """Randomly shuffles pairs and labels together to maintain correspondence."""
    if seed is not None:
        np.random.seed(seed)
    pairs_arr = np.array(pairs)
    labels_arr = np.array(labels)
    perm = np.random.permutation(len(pairs))
    return pairs_arr[perm], labels_arr[perm]


def sample_negative_pairs(
    split_edges: pd.DataFrame,
    ft_edges: pd.DataFrame,
    embeddings: torch.Tensor,
    split_name: str,
    seed: int,
    negative_ratio: float = 1.0,
) -> FineTuneDataset:
    """
    Build a fine-tuning dataset with positive and negative pairs.

    Args:
        split_edges: Edges for a specific split (train/val/test)
        ft_edges: All positive edges of this type
        embeddings: Node embeddings tensor
        split_name: Name of split for logging
        seed: Random seed
        negative_ratio: Ratio of negative to positive samples

    Returns:
        Dataset with positive and negative pairs
    """
    pos_pairs = list(zip(split_edges["x_index"], split_edges["y_index"], strict=False))
    num_pos = len(pos_pairs)
    _logger.debug(f"Number of positive {split_name} pairs: {num_pos}")

    pos_pairs_set = set(zip(ft_edges["x_index"], ft_edges["y_index"], strict=False))
    src_nodes = ft_edges["x_index"].unique()
    dst_nodes = ft_edges["y_index"].unique()

    np.random.seed(seed)
    neg_pairs = set()
    while len(neg_pairs) < num_pos * negative_ratio:
        pair = (np.random.choice(src_nodes), np.random.choice(dst_nodes))
        if pair not in pos_pairs_set:
            neg_pairs.add(pair)
    neg_pairs = list(neg_pairs)
    _logger.info(f"Number of negative {split_name} pairs: {len(neg_pairs)}")

    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * num_pos + [0] * len(neg_pairs)

    all_pairs, all_labels = scramble_pairs_and_labels(all_pairs, all_labels, seed)

    return FineTuneDataset(all_pairs, all_labels, embeddings)


def build_dataset_from_edges(
    x_type: str, relation: str, y_type: str, kg_edges_split: pd.DataFrame, embeddings: torch.Tensor
) -> tuple[FineTuneDataset, FineTuneDataset, FineTuneDataset]:
    """Build train/val/test datasets for a specific edge type."""

    # Filter edges
    ft_edges = kg_edges_split[
        (kg_edges_split["x_type"] == x_type)
        & (kg_edges_split["relation"] == relation)
        & (kg_edges_split["y_type"] == y_type)
    ].copy()

    datasets = []
    for split_id, split_name in enumerate(["train", "val", "test"]):
        split_edges = ft_edges[ft_edges["split"] == split_id]
        seed = conf.seed + (split_id * 100)
        dataset = sample_negative_pairs(
            split_edges,
            ft_edges,
            embeddings,
            split_name,
            seed,
            conf.finetune.negative_ratio,
        )
        datasets.append(dataset)

    return tuple(datasets)


def build_disease_gene_dataset(
    kg_edges_split: pd.DataFrame,
    embeddings: torch.Tensor,
) -> tuple[FineTuneDataset, FineTuneDataset, FineTuneDataset]:
    return build_dataset_from_edges("disease", "disease_protein", "gene/protein", kg_edges_split, embeddings)


def build_drug_effect_dataset(
    kg_edges_split: pd.DataFrame,
    embeddings: torch.Tensor,
) -> tuple[FineTuneDataset, FineTuneDataset, FineTuneDataset]:
    return build_dataset_from_edges("drug", "drug_effect", "effect/phenotype", kg_edges_split, embeddings)


def build_drug_disease_dataset(
    kg_edges_split: pd.DataFrame,
    embeddings: torch.Tensor,
) -> tuple[FineTuneDataset, FineTuneDataset, FineTuneDataset]:
    """Build train/val/test datasets for drug-disease edges (indications, contraindications, off-label uses)."""

    # Filter edges and assign labels
    ft_edges = kg_edges_split[
        (kg_edges_split["x_type"] == "disease")
        & (kg_edges_split["relation"].isin(["indication", "contraindication", "off_label_use"]))
        & (kg_edges_split["y_type"] == "drug")
    ].copy()

    ft_edges["label"] = ft_edges["relation"].map({"indication": 1, "off_label_use": 1, "contraindication": 0})

    datasets = []
    for split_id, split_name in enumerate(["train", "val", "test"]):
        split_edges = ft_edges[ft_edges["split"] == split_id]
        _logger.debug(f"{split_name} edges: {split_edges['relation'].value_counts()}")

        pairs = list(zip(split_edges["x_index"], split_edges["y_index"], strict=False))
        labels = list(split_edges["label"].values)
        pairs, labels = scramble_pairs_and_labels(pairs, labels, conf.seed)

        datasets.append(FineTuneDataset(pairs, labels, embeddings))

    return tuple(datasets)


def build_PD_disease_gene_dataset(
    kg_edges_split: pd.DataFrame,
    embeddings: torch.Tensor,
) -> FineTuneDataset:
    """Build a fine-tuning dataset for PD disease-gene edges from expert curation and experimental evidence."""

    # Read KG nodes and get gene nodes
    kg_nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    gene_nodes = kg_nodes[kg_nodes["node_type"] == "gene/protein"].copy()

    # Get PD GWAS/RVAS genes (positives)
    eval_lists = pd.ExcelFile(DATA_DIR / "ASAP" / "gene_lists" / "PD_related_genelists_022025.xlsx")
    pd_gwas = pd.read_excel(eval_lists, sheet_name="PD_genes_GWAS_RVAS", header=None, names=["gene", "source"])
    pd_genes = gene_nodes[gene_nodes["node_name"].isin(pd_gwas["gene"].unique())]

    # Get protein coding genes not in gRNA lists (negatives)
    hgnc = pd.read_csv(conf.paths.mappings.hgnc_path, sep="\t", low_memory=False)
    protein_coding = hgnc[hgnc["Locus group"] == "protein-coding gene"]["Approved symbol"]

    gRNA_nodes_path = DATA_DIR / "ASAP" / "gene_lists" / "gRNA_nodes_02172025.xlsx"
    gRNA_PCSF = pd.read_excel(gRNA_nodes_path, sheet_name="union of four lists")
    neighbors = pd.read_excel(gRNA_nodes_path, sheet_name="1-hop neighbors (TNet v2_0.2)")
    gRNA_genes = pd.concat([gRNA_PCSF, neighbors])["GeneName"]

    pc_negative_genes = gene_nodes[
        gene_nodes["node_name"].isin(protein_coding) & ~gene_nodes["node_name"].isin(gRNA_genes)
    ]

    # Get PD node index and create pairs
    PD_index = kg_nodes[kg_nodes["node_name"] == "Parkinson disease"]["node_index"].values[0]
    pos_pairs = list(zip([PD_index] * len(pd_genes), pd_genes["node_index"], strict=False))
    neg_pairs = list(zip([PD_index] * len(pc_negative_genes), pc_negative_genes["node_index"], strict=False))

    # Create labels and combine
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    # Shuffle and build dataset
    all_pairs, all_labels = scramble_pairs_and_labels(all_pairs, all_labels, conf.seed)
    return FineTuneDataset(all_pairs, all_labels, embeddings)


FINETUNE_DATASETS = {
    "disease-gene": build_disease_gene_dataset,
    "drug-disease": build_drug_disease_dataset,
    "drug-effect": build_drug_effect_dataset,
    "PD-disease-gene": build_PD_disease_gene_dataset,
}
