import csv
import json
import logging
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import typer
from huggingface_hub import snapshot_download

from src.config import conf
from src.constants import TORCH_DEVICE

_logger = logging.getLogger(__name__)
cli = typer.Typer(help="Main entry point for the CLI.")

USER_AGENT = "Mozilla/5.0 (compatible; NeuroKGDownloader/1.0; +https://example.org)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}


@cli.callback()
def callback() -> None:
    """Inject the logging configuration into the logging system."""
    import logging.config

    from src.config import conf

    logging.config.dictConfig(conf.logging.model_dump())

    _logger.debug(f"Using '{TORCH_DEVICE}' as the PyTorch device.")


@cli.command()
def download_proton(
    download_splits: bool = typer.Option(False, "--download_splits", help="Download the disease_splits folder"),
) -> None:
    _logger.info("Downloading model from Hugging Face...")

    repo_id = conf.proton.huggingface_repository
    checkpoint_dir = conf.paths.checkpoint.base_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Created checkpoint directory: %s", checkpoint_dir)

    try:
        _logger.info("Downloading files from %s...", repo_id)
        ignore_patterns = ["*.md"]
        if not download_splits:
            ignore_patterns.append("disease_splits/**")
            _logger.info("Excluding disease_splits folder from download")
        snapshot_download(repo_id=repo_id, local_dir=checkpoint_dir, ignore_patterns=ignore_patterns)
        _logger.info("Successfully downloaded all files to %s", checkpoint_dir)
    except Exception:
        _logger.exception("Failed to download model files")
        raise


def _download_from_dataverse(
    base_url: str,
    neurokg_dir: Path,
    file_id: str,
    target_filename: str,
    is_zip: bool = False,
    is_tab: bool = False,
) -> Path:
    download_url = f"{base_url}/access/datafile/{file_id}"
    temp_path = neurokg_dir / f"{target_filename}.tmp"
    save_path = neurokg_dir / target_filename

    _logger.info("Downloading %s from %s...", target_filename, download_url)
    request = Request(download_url, headers=DEFAULT_HEADERS)  # noqa: S310
    with urlopen(request) as response, open(temp_path, "wb") as out_file:  # noqa: S310
        out_file.write(response.read())

    if is_zip:
        _logger.info("Extracting %s from zip archive...", target_filename)
        with (
            zipfile.ZipFile(temp_path, "r") as zip_ref,
            zip_ref.open(target_filename) as source,
            open(save_path, "wb") as target,
        ):
            target.write(source.read())
        temp_path.unlink()
    elif is_tab:
        _logger.info("Converting tab-delimited %s to CSV...", target_filename)
        with (
            open(temp_path, encoding="utf-8") as tab_file,
            open(save_path, "w", encoding="utf-8", newline="") as csv_file,
        ):
            tab_reader = csv.reader(tab_file, delimiter="\t")
            csv_writer = csv.writer(csv_file)
            for row in tab_reader:
                csv_writer.writerow(row)
        temp_path.unlink()
    else:
        temp_path.rename(save_path)

    _logger.info("Successfully saved %s to %s", target_filename, save_path)
    return save_path


@cli.command()
def download_neurokg() -> None:
    _logger.info("Downloading NeuroKG files from Harvard Dataverse...")

    base_url = conf.neurokg.dataverse_base_url
    persistent_id = conf.neurokg.dataverse_persistent_id
    neurokg_dir = conf.paths.kg.base_dir

    neurokg_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Created NeuroKG directory: %s", neurokg_dir)

    try:
        dataset_url = f"{base_url}/datasets/:persistentId?persistentId={persistent_id}"
        _logger.info("Fetching dataset information from %s...", dataset_url)
        request = Request(dataset_url, headers=DEFAULT_HEADERS)  # noqa: S310

        with urlopen(request) as response:  # noqa: S310
            dataset_info = json.loads(response.read())

        found_edges = found_nodes = False

        for file_info in dataset_info.get("data", {}).get("latestVersion", {}).get("files", []):
            data_file = file_info.get("dataFile", {})
            filename = data_file.get("filename", "")
            original_filename = data_file.get("originalFileName", "")
            file_id = data_file.get("id")

            if filename == "edges.csv.zip":
                _logger.info("Found %s with file ID: %s", filename, file_id)
                _download_from_dataverse(base_url, neurokg_dir, file_id, "edges.csv", is_zip=True)
                found_edges = True

            elif filename == "nodes.tab" or original_filename == "nodes.csv":
                _logger.info("Found %s (original: %s) with file ID: %s", filename, original_filename, file_id)
                _download_from_dataverse(base_url, neurokg_dir, file_id, "nodes.csv", is_tab=True)
                found_nodes = True

        if not (found_edges or found_nodes):
            _logger.warning("No edges or nodes files found in the dataset")
        else:
            _logger.info("Successfully downloaded and processed NeuroKG files to %s", neurokg_dir)

    except Exception:
        _logger.exception("Failed to download NeuroKG files")
        raise


@cli.command()
def train(
    save_embeddings: bool = typer.Option(False, "--save-embeddings", help="Save embeddings"),
    save_relation_weights: bool = typer.Option(False, "--save-relation-weights", help="Save relation weights"),
) -> None:
    _logger.info("Training...")
    from src.train import run_train

    if save_embeddings:
        conf.proton.training.output_options.save_embeddings = True
    if save_relation_weights:
        conf.proton.training.output_options.save_relation_weights = True

    run_train()


@cli.command()
def split() -> None:
    _logger.info("Splitting data...")
    from src.splits import run_split

    run_split()


@cli.command()
def explain() -> None:
    _logger.info("Explaining...")
    from src.explainer import run_explainer

    run_explainer()


@cli.command()
def cosine_similarity() -> None:
    _logger.info("Computing cosine similarity from embeddings...")
    from src.embeddings import get_cosine_similarity

    get_cosine_similarity()


@cli.command()
def log_conf() -> None:
    """Log the configuration."""
    _logger.info(conf)


@cli.command("random-walks")
def random_walks(
    nodes_file: Path = typer.Argument(..., help="Path to the nodes CSV file"),  # noqa: B008
    edges_file: Path = typer.Argument(..., help="Path to the edges CSV file"),  # noqa: B008
    start: int = typer.Option(..., "--start", "-s", help="ID (Index) of the starting node"),
    length: int = typer.Option(..., "--length", "-l", help="Length of the walk"),
    num_walks: int = typer.Option(10000, "--num-walks", "-n", help="Number of random walks to simulate"),
) -> None:
    from src.baselines import perform_random_walks

    perform_random_walks(nodes_file, edges_file, start, length, num_walks)
