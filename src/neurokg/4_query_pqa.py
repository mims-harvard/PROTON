"""
QUERY PQA
This script contains the main function for querying the PQA.

Requires `conda activate PaperQA` with pqapi version 7.2.0 on Python 3.11.11.
Also set the PQA_API_KEY environment variable in .env file.

Run with:
```
conda activate PaperQA
cd src/neurokg
python 4_query_PQA.py submit
python 4_query_PQA.py submit --pqa-random
python 4_query_PQA.py retrieve --pqa-id <path_to_results_directory>
For example:
    python 4_query_PQA.py retrieve --pqa-id 2025-02-17_01-55-29
    python 4_query_PQA.py submit --pqa-random
    python 4_query_PQA.py retrieve --pqa-id 2025-02-18_04-30-27_random
```
"""

import asyncio
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pqapi
import typer
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import conf

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(conf.paths.secrets_path)

# Create typer app
app = typer.Typer(help="Submit or retrieve PQA queries.")


def submit_queries(hparams):
    # Directory to save results
    pqa_prompts_dir = conf.paths.pqa_prompts_dir
    pqa_results_dir = conf.paths.pqa_results_dir

    # Get date and time
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if hparams["pqa_random"]:
        date_time = date_time + "_random"
    pqa_prompts_dir = pqa_prompts_dir / date_time
    pqa_results_dir = pqa_results_dir / date_time

    # Create directories
    pqa_prompts_dir.mkdir(parents=True, exist_ok=True)
    pqa_results_dir.mkdir(parents=True, exist_ok=True)

    # Read KG nodes and edges
    logger.info("Reading KG nodes and edges...")
    kg_nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    kg_edges = pd.read_csv(
        conf.paths.kg.edges_path, dtype={"edge_index": int, "x_index": int, "y_index": int}, low_memory=False
    )

    ########################################
    # CREATE PROMPTS
    ########################################

    # Select diseases of interest
    diseases_for_prompts = [36436, 39348, 41591, 39579, 39528, 43130]

    # Print nodes and calculate degree
    for disease in diseases_for_prompts:
        node_name = kg_nodes.loc[kg_nodes["node_index"] == disease, "node_name"].values[0]
        edge_count = len(kg_edges[kg_edges["x_index"] == disease])
        logger.info(f"- {node_name} ({disease}), {edge_count} edges")

    # Define template text
    template_text = """Is there any scientific or medical evidence to support an association between {0} and {1}? Please rate the strength of the evidence on a 5-point scale, where:
    1 = No evidence (zero papers mentioning both {0} and {1})
    2 = Weak evidence (1-2 papers mentioning both {0} and {1} and no experimental evidence)
    3 = Moderate evidence (3-4 papers mentioning both {0} and {1} or experimental evidence)
    4 = Strong evidence (5-6 papers mentioning both {0} and {1} or several experimental studies)
    5 = Very strong evidence (more than 6 papers mentioning both {0} and {1} or substantial experimental evidence)
    In your response, please also explain the reasoning behind your rating and reference any relevant scientific or medical sources (e.g., peer-reviewed studies, clinical guidelines, experimental data) that support your assessment. For each part of your response, indicate which sources most support it via citation keys at the end of sentences, like (Example2012Example pages 3-4). Only use valid citation keys.

    Instructions to the LLM: Respond with the following XML format exactly.
    <response>
    <reasoning>...</reasoning>
    <rating>...</rating>
    </response>

    `rating` is one of the following (must match exactly): 1, 2, 3, 4, or 5. Do not include any additional keys or text."""

    # Generate disease prompts
    disease_prompts = []

    for disease in tqdm(diseases_for_prompts, desc="Generating prompts", total=len(diseases_for_prompts)):
        if hparams["pqa_random"]:
            # Disease neighborhood
            neighborhood = kg_edges[kg_edges["x_index"] == disease][["x_index", "x_name", "y_index", "y_name"]].copy()

            # Get list of nodes NOT in neighborhood
            not_neighborhood = set(kg_nodes["node_index"].tolist()) - set(neighborhood["y_index"].tolist())
            not_neighborhood_df = kg_nodes[kg_nodes["node_index"].isin(not_neighborhood)]

            # Subset to node types in "gene/protein", "disease", "drug", "exposure"
            edge_types = ["gene/protein", "disease", "drug", "exposure"]
            not_neighborhood_df = not_neighborhood_df[not_neighborhood_df["node_type"].isin(edge_types)]

            # Randomly select 100 rows from not_neighborhood_df
            random_nodes = not_neighborhood_df.sample(n=100)

            # Construct edge data frame
            disease_name = kg_nodes.loc[kg_nodes["node_index"] == disease, "node_name"].values[0]
            disease_edges = pd.DataFrame({
                "x_index": [disease] * len(random_nodes),  # Repeat disease ID
                "x_name": [disease_name] * len(random_nodes),  # Repeat disease name
                "y_index": random_nodes["node_index"].values,  # Randomly selected node indices
                "y_name": random_nodes["node_name"].values,  # Corresponding names
            })

        else:
            disease_edges = kg_edges[kg_edges["x_index"] == disease][["x_index", "x_name", "y_index", "y_name"]].copy()

        # Construct prompt
        disease_edges["prompt"] = disease_edges.apply(
            lambda row: template_text.format(row["x_name"], row["y_name"]), axis=1
        )

        disease_edges.to_csv(pqa_prompts_dir / f"PQA_prompts_{disease}.csv", index=False, encoding="utf-8")
        disease_prompts.append(disease_edges)

    # Concatenate all prompts
    disease_prompts_df = pd.concat(disease_prompts, ignore_index=True)

    # Save combined file
    disease_prompts_df.to_csv(pqa_prompts_dir / "PQA_prompts.csv", index=False, encoding="utf-8")

    ########################################
    # SUBMIT JOBS
    ########################################

    # disease_prompts_df = disease_prompts_df.head(5)
    queries = disease_prompts_df["prompt"].tolist()

    # Submit jobs
    logger.info(f"Submitting {len(queries)} jobs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    jobs = [pqapi.submit_agent_job(query=q) for q in queries]
    logger.info(f"Submitted {len(jobs)} jobs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    # Add query IDs to jobs df
    query_ids = [job["metadata"]["query_id"] for job in jobs]
    disease_prompts_df["query_id"] = query_ids
    disease_prompts_df.to_csv(pqa_prompts_dir / "PQA_prompts_with_ids.csv", index=False, encoding="utf-8")

    # Save jobs to file
    with open(pqa_results_dir / "jobs.pkl", "wb") as f:
        pickle.dump(jobs, f)


async def process_batch(ids):
    """Process a batch of query IDs to retrieve results."""
    tasks = [pqapi.async_get_query(query_id) for query_id in ids]
    results = await asyncio.gather(*tasks)
    return results


async def retrieve_results(hparams):
    # If no results directory is provided, throw an error
    if hparams["pqa_id"] is None:
        raise ValueError("No PQA ID provided, cannot retrieve results.")
    pqa_prompts_dir = conf.paths.pqa_prompts_dir / hparams["pqa_id"]
    pqa_results_dir = conf.paths.pqa_results_dir / hparams["pqa_id"]

    # Check if prompts and results directories exist
    if not pqa_prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory {pqa_prompts_dir} does not exist.")
    if not pqa_results_dir.exists():
        raise FileNotFoundError(f"Results directory {pqa_results_dir} does not exist.")

    # Load jobs and prompts
    disease_prompts_df = pd.read_csv(pqa_prompts_dir / "PQA_prompts_with_ids.csv", dtype={"query_id": str})

    # Get query IDs
    query_ids = disease_prompts_df["query_id"].tolist()
    logger.info(f"Number of queries: {len(query_ids)}")

    BATCH_SIZE = 25
    results = []

    for i in tqdm(range(0, len(query_ids), BATCH_SIZE), desc="Retrieving results", total=len(query_ids) // BATCH_SIZE):
        batch_ids = query_ids[i : i + BATCH_SIZE]
        batch_results = await process_batch(batch_ids)
        results.extend(batch_results)

    return results, disease_prompts_df


def retrieve_and_save_results(hparams):
    # Retrieve results
    results, disease_prompts_df = asyncio.run(retrieve_results(hparams))

    # Get results directory
    pqa_results_dir = conf.paths.pqa_results_dir / hparams["pqa_id"]

    # Save results to file
    with open(pqa_results_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Retrieve responses
    responses = [(result["id"], result["response"]["answer"]["answer"]) for result in results if result is not None]
    responses_df = pd.DataFrame(responses, columns=["query_id", "response"])

    # Print shape of responses
    logger.info(f"Number of responses: {len(responses)}")

    # Merge responses with prompts
    results_df = pd.merge(disease_prompts_df, responses_df, on="query_id", how="left")

    # Save results to CSV
    results_df.to_csv(pqa_results_dir / "PQA_results.csv", index=False, encoding="utf-8")


@app.command("submit")
def submit(
    pqa_random: bool = typer.Option(False, "--pqa-random", help="Randomly select 200 edges for PQA queries."),
) -> None:
    """Submit queries to PQA."""
    # Set seed
    np.random.seed(conf.seed)

    hparams = {
        "mode": "submit",
        "pqa_random": pqa_random,
    }
    submit_queries(hparams)


@app.command("retrieve")
def retrieve(
    pqa_id: str = typer.Option(..., "--pqa-id", help="Name of directory containing the PQA jobs."),
) -> None:
    """Retrieve results from PQA."""
    # Set seed
    np.random.seed(conf.seed)

    hparams = {
        "mode": "retrieve",
        "pqa_id": pqa_id,
    }
    retrieve_and_save_results(hparams)


if __name__ == "__main__":
    app()
