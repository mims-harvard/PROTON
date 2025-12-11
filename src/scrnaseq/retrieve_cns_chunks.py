"""
DATA RETRIEVAL SCRIPT
This script loads the nervous subset of the CZ CELLxGENE Census dataset into a single AnnData object,
then writes that object to disk for downstream analysis.
"""

# import project config file
import sys
import time

# import CELLxGENE
import cellxgene_census
import pandas as pd
import tiledbsoma
from anndata import AnnData
from scipy import sparse

sys.path.append("../../..")
import project_config  # ty: ignore[unresolved-import]

# LOAD IN CHUNKS

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
    # get human data
    human = census["census_data"]["homo_sapiens"]

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

        # initialize lazy query
        query = human.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(soma_ids,)),
        )

        # get cell metadata (obs) and gene metadata (var)
        query_obs = query.obs().concat().to_pandas()
        query_var = query.var().concat().to_pandas()

        # set indices of metadata
        query_obs = query_obs.set_index("soma_joinid")
        query_var = query_var.set_index("soma_joinid")

        # save to CSV
        query_obs.to_csv(project_config.CELLXGENE_DIR / "chunks" / (group_id + "_obs.csv"))
        query_var.to_csv(project_config.CELLXGENE_DIR / "chunks" / (group_id + "_var.csv"))

        # to get actual data, get iterator for X
        iterator = query.X("raw").tables()
        # indexer = query.indexer

        # iterate in chunks
        for index, chunk in enumerate(iterator):
            # record time
            start_time = time.time()

            # each chunk is a sparse matrix, so zeroes are not stored in the data object
            # convert to pandas
            chunk_df = chunk.to_pandas(split_blocks=True, self_destruct=True)
            del chunk  # delete chunk to save memory

            # obs and var indices are the sparse indices of the chunk
            # chunk_data is the sparse data of the chunk
            chunk_data = chunk_df["soma_data"].astype(int)  # convert to int, as counts data was stored as float
            cell_indices = chunk_df["soma_dim_0"]  # obs values
            gene_indices = chunk_df["soma_dim_1"]  # var values
            # del chunk_df # delete chunk data frame to save memory

            # get total number of cells and remap IDs within chunk
            unique_cell_indices = sorted(cell_indices.unique())  # note! this changes the order of the cells
            n_cells = len(unique_cell_indices)
            cell_index_dict = {unique_cell_indices[i]: i for i in range(n_cells)}
            new_cell_indices = cell_indices.map(cell_index_dict)

            # subset cell metadata
            chunk_cell_metadata = query_obs.loc[unique_cell_indices]

            # get total number of genes
            n_genes = query_var.shape[0]

            # get CSR matrix
            # to convert back, run pd.DataFrame.sparse.from_spmatrix(csr)
            print("- Chunk " + str(index) + ": ")
            print("   - Number of cells: " + str(n_cells))
            print("   - Number of genes: " + str(n_genes))
            print(f"   - Cell {min(unique_cell_indices)} to {max(unique_cell_indices)}")
            csr = sparse.csr_matrix((chunk_data, (new_cell_indices, gene_indices)), shape=(n_cells, n_genes))

            # convert to dense matrix and save to CSV
            # dense_mat = csr.todense()
            # dense_mat.tofile(project_config.CELLXGENE_DIR / 'chunks' / (group_id + "_" + str(index) + '_matrix.csv'), sep = ',')
            chunk_df.to_csv(
                project_config.CELLXGENE_DIR / "chunks" / (group_id + "_" + str(index) + "_matrix.csv"), index=False
            )

            # convert to AnnData object
            adata = AnnData(X=csr, obs=chunk_cell_metadata)

            # write CSR and metadata to disk
            sparse.save_npz(
                project_config.CELLXGENE_DIR / "chunks" / (group_id + "_" + str(index) + "_matrix.npz"), csr
            )
            chunk_cell_metadata.to_csv(
                project_config.CELLXGENE_DIR / "chunks" / (group_id + "_" + str(index) + "_metadata.csv"), index=False
            )
            adata.write_h5ad(project_config.CELLXGENE_DIR / "chunks" / (group_id + "_" + str(index) + "_adata.h5ad"))

            # record time
            end_time = time.time()

            # print time
            print("   - Took " + str(end_time - start_time) + " seconds")

            # if needed, break
            # break

    # close the query
    query.close()

# print completion message
print("All chunks processed!")
