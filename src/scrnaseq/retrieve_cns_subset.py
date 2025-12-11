"""
DATA RETRIEVAL SCRIPT
This script loads the nervous subset of the CZ CELLxGENE Census dataset into a single AnnData object,
then writes that object to disk for downstream analysis.
"""

import sys

import cellxgene_census

sys.path.append("../../..")
import project_config  # ty: ignore[unresolved-import]

# define URI path
uri_path = str(project_config.CELLXGENE_DATASET / "census")

# open census
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

    # save to CSV
    nervous_metadata.to_csv(project_config.CELLXGENE_DIR / "nervous_metadata.csv", index=False)

    # get all data from brain slice as AnnData object
    nervous_subset = cellxgene_census.get_anndata(
        census,
        "Homo sapiens",
        obs_value_filter="is_primary_data == True and tissue_general in ['brain', 'spinal cord']",
    )

    # save AnnData object to file
    nervous_subset.write(str(project_config.CELLXGENE_DIR / "nervous_subset.h5ad"))
