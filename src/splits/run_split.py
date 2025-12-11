import logging

import pandas as pd

from .utils import (
    create_and_save_summary,
    create_disease_groups,
    load_data,
    preprocess_diseases,
    save_splits,
)

_logger = logging.getLogger(__name__)


def run_split():
    """Run the disease-based splitting process."""
    nodes, edges = load_data()
    _logger.info(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")
    disease_nodes, embeds = preprocess_diseases(nodes)
    _logger.info(f"Generated {embeds.shape[0]} embeddings for {len(disease_nodes)} diseases.")
    disease_groups = create_disease_groups(disease_nodes, edges, embeds)

    if not disease_groups:
        _logger.info("No disease groups created. Exiting.")
        return

    _logger.info(f"Created {len(disease_groups)} disease groups.")

    disease_splits = pd.concat(list(disease_groups.values()), axis=0).reset_index(drop=True)
    disease_splits, edge_count, all_split_edges = save_splits(disease_splits, edges)

    if disease_splits.empty:
        _logger.info("No splits with edges found. Exiting.")
        return

    _logger.info(f"Saved {len(disease_splits)} disease splits with {len(all_split_edges)} edges.")

    create_and_save_summary(disease_splits, all_split_edges, edge_count)
