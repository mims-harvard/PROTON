"""
DATA RETRIEVAL SCRIPT
This script loads the nervous subset of the CZ CELLxGENE Census dataset into a single AnnData object,
then writes that object to disk for downstream analysis.
"""

# import project config file
import sys

import cellxgene_census
import pandas as pd

sys.path.append("../../..")
import project_config  # ty: ignore[unresolved-import]

# LOAD GROUPS

# define URI path
uri_path = str(project_config.CELLXGENE_DATASET / "census")

# open local version of soma
print("Opening census at " + uri_path + "...")
with cellxgene_census.open_soma(uri=uri_path) as census:
    # get metadata for cells in brain and spinal cord
    # remove duplicate cells present across multiple datasets with is_primary_data == True
    # for brain data, reduces from approx. 16M to 9M with is_primary_data == True
    nervous_metadata = (
        census["census_data"]["homo_sapiens"]
        .obs.read(value_filter="is_primary_data == True and tissue_general in ['brain', 'spinal cord']")
        .concat()
        .to_pandas()
    )

    # group by tissue, cell type, and dataset
    nervous_groups = (
        nervous_metadata.groupby([
            "tissue",
            "tissue_ontology_term_id",
            "cell_type",
            "cell_type_ontology_term_id",
            "dataset_id",
        ])
        .size()
        .reset_index(name="cell_counts")
    )
    nervous_groups = nervous_groups.sort_values("cell_counts", ascending=False)
    nervous_groups = nervous_groups[nervous_groups["cell_counts"] > 1000]

    # read all datasets in the Census
    census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    census_datasets = census_datasets.set_index("dataset_id")

    # add dataset data to nervous groups data frame
    nervous_groups = pd.merge(nervous_groups, census_datasets, on="dataset_id")
    nervous_groups = nervous_groups.sort_values("cell_counts", ascending=False)

    # create new ID column: stripped UBERON ID + stripped cell type ID + dataset ID
    nervous_groups["group_id"] = (
        nervous_groups["tissue_ontology_term_id"].str.replace("UBERON:", "")
        + "_"
        + nervous_groups["cell_type_ontology_term_id"].str.replace("CL:", "")
        + "_"
        + nervous_groups["dataset_id"]
    )

    # save to CSV
    nervous_groups.to_csv(project_config.CELLXGENE_DIR / "nervous_groups.csv", index=False)


# open local version of soma
print("Opening census at " + uri_path + "...")
with cellxgene_census.open_soma(uri=uri_path) as census:
    # iterate over possible combinations
    for _group_index, row in nervous_groups.iterrows():
        # get group ID
        group_id = row["group_id"]
        print("Processing group " + group_id + "...")
        print(" - Tissue: " + row["tissue"])
        print(" - Cell Type: " + row["cell_type"])
        print(" - Dataset: " + row["dataset_id"])

        # get soma IDs
        group_subset = nervous_metadata[
            (nervous_metadata["tissue"] == row["tissue"])
            & (nervous_metadata["cell_type"] == row["cell_type"])
            & (nervous_metadata["dataset_id"] == row["dataset_id"])
        ]
        soma_ids = group_subset["soma_joinid"].to_list()

        # get all data from brain slice as AnnData object
        group_subset = cellxgene_census.get_anndata(census, "Homo sapiens", obs_coords=soma_ids)

        # write to file
        group_subset.write(str(project_config.CELLXGENE_DIR / "no_chunks" / (group_id + ".h5ad")))

# print completion message
print("All groups processed!")
