import logging
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.config import conf
from src.constants import FINETUNING_DIR
from src.dataloaders import load_graph, partition_graph
from src.models import FineTuneAdapter

from .generate_data import (
    FINETUNE_DATASETS,
    FineTuneDataset,
)

_logger = logging.getLogger(__name__)


def load_pretrained_embeddings(best_ckpt: str) -> torch.Tensor:
    """Load pre-trained embeddings from disk, optionally randomizing them.

    Args:
        best_ckpt: Checkpoint filename to load embeddings from

    Returns:
        torch.Tensor: Node embeddings [num_nodes, embed_dim]
    """
    embed_path = conf.paths.checkpoint.embeddings_path.parent / best_ckpt.replace(".ckpt", "_embeddings.pt")

    if not embed_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embed_path}.")

    _logger.debug(f"Loading embeddings from {embed_path}...")
    embeddings = torch.load(embed_path)

    if conf.finetune.random_baseline:
        embeddings = torch.randn_like(embeddings)
        _logger.debug("Using random baseline embeddings...")

    return embeddings


def get_edge_partitions() -> pd.DataFrame:
    """Get train/val/test edge partitions from pretraining for fine-tuning."""
    pl.seed_everything(conf.seed, workers=True)

    _, _, test_kg = partition_graph(load_graph())

    edges = []
    for etype in test_kg.canonical_etypes:
        src, dst = test_kg.edges(etype=etype)

        edges.append(
            pd.DataFrame({
                "relation": etype[1],
                "x_index": test_kg.nodes[etype[0]].data["node_index"][src].numpy(),
                "x_type": etype[0],
                "y_index": test_kg.nodes[etype[2]].data["node_index"][dst].numpy(),
                "y_type": etype[2],
                "split": test_kg.edges[etype].data["mask"].numpy().astype(int),
            })
        )

    return pd.concat(edges).sort_values(["x_index", "y_index", "relation"]).reset_index(drop=True)


def create_data_loaders(
    train_data: FineTuneDataset,
    val_data: FineTuneDataset,
    test_data: FineTuneDataset,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for training, validation and test datasets.

    Args:
        train_data: Training dataset containing node pairs and labels
        val_data: Validation dataset containing node pairs and labels
        test_data: Test dataset containing node pairs and labels
        batch_size: Number of samples per batch. Defaults to 512.
        num_workers: Number of subprocesses for data loading. Defaults to 4.

    Returns:
        A tuple of (train_loader, val_loader, test_loader) where:
            - train_loader shuffles data and iterates over training samples
            - val_loader iterates over validation samples sequentially
            - test_loader iterates over test samples sequentially
    """
    loader_args = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train_data, shuffle=True, **loader_args),
        DataLoader(val_data, shuffle=False, **loader_args),
        DataLoader(test_data, shuffle=False, **loader_args),
    )


def train_adapter(
    model: FineTuneAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 10,
) -> tuple[pl.Trainer, str, str]:
    """
    Train the fine-tune model using PyTorch Lightning.

    Args:
        model: MLP-based classifier
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum number of training epochs

    Returns:
        Tuple of (trainer, run_id, run_name)
    """
    curr_time = datetime.now()
    run_name = f"{conf.finetune.dataset} at {curr_time.strftime('%H:%M:%S on %m/%d/%Y')}"
    run_id = curr_time.strftime("%Y_%m_%d_%H_%M_%S") + f"_{conf.finetune.dataset}"

    if conf.finetune.random_baseline:
        run_id += "_random"
        run_name += " (random)"

    wandb_logger = WandbLogger(
        name=run_name,
        project=conf.wandb.finetune_project_name,
        entity=conf.wandb.entity_name,
        save_dir=FINETUNING_DIR,
        id=run_id,
        resume="never",
    )
    wandb_logger.watch(model, log="all")

    callbacks = [
        ModelCheckpoint(
            monitor="val/auroc",
            dirpath=FINETUNING_DIR / "checkpoints",
            filename=run_id + "_{epoch}-{step}",
            save_top_k=1,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    train_limits = {
        "limit_train_batches": 5 if conf.proton.training.output_options.debug else 1.0,
        "limit_val_batches": 1 if conf.proton.training.output_options.debug else 1.0,
        "max_epochs": 3 if conf.proton.training.output_options.debug else max_epochs,
    }

    if conf.proton.training.output_options.debug:
        _logger.debug("Debug mode enabled.")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1 if conf.proton.training.output_options.debug else 10,
        val_check_interval=0.25,
        deterministic=True,
        **train_limits,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer, run_id, run_name


def test_adapter(model: FineTuneAdapter, trainer: pl.Trainer, test_loader: DataLoader) -> dict:
    """Tests the model on the test dataset."""
    return trainer.test(model, dataloaders=test_loader)  # ty: ignore[invalid-return-type]


def train_PD_adapter(
    base_trainer: pl.Trainer,
    model: FineTuneAdapter,
    embeddings: torch.Tensor,
    kg_edges_split: pd.DataFrame,
    run_id: str,
    run_name: str,
) -> tuple[pl.Trainer, str, str]:
    """Fine-tunes the model on PD-specific disease-gene edges."""
    _logger.debug("Fine-tuning on PD-related genes...")

    pd_dataset = FINETUNE_DATASETS["PD-disease-gene"](kg_edges_split, embeddings)
    pd_loader = DataLoader(pd_dataset, batch_size=512, shuffle=True, num_workers=1)

    pd_run_id = f"{run_id}_PD"
    pd_wandb_logger = WandbLogger(
        name=f"{run_name} (PD)",
        project=conf.wandb.finetune_project_name,
        entity=conf.wandb.entity_name,
        save_dir=FINETUNING_DIR,
        id=pd_run_id,
        resume="never",
    )

    pd_trainer = pl.Trainer(
        max_epochs=100,
        accelerator=base_trainer.accelerator,
        devices=base_trainer.num_devices,
        logger=pd_wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="train/auroc",
                dirpath=FINETUNING_DIR / "checkpoints",
                filename=f"{pd_run_id}_{{epoch}}-{{step}}",
                save_top_k=1,
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=1 if conf.proton.training.output_options.debug else 10,
        deterministic=True,
    )

    pd_trainer.fit(model, pd_loader)

    return pd_trainer, pd_run_id, f"{run_name} (PD)"


def finetune():
    """Fine-tune a pre-trained model on a specific dataset."""
    embeddings = load_pretrained_embeddings(conf.paths.checkpoint.checkpoint_path.name)
    kg_edges_split = get_edge_partitions()

    conf.finetune.negative_ratio = 3.0 if conf.finetune.dataset == "disease-gene" else 1.0  # ty: ignore[invalid-assignment]

    train_data, val_data, test_data = FINETUNE_DATASETS[conf.finetune.dataset](kg_edges_split, embeddings)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=512, num_workers=1
    )

    embed_dim = embeddings.shape[1]
    finetune_adapter = FineTuneAdapter(input_dim=2 * embed_dim, hidden_dim=128, lr=1e-3)
    trainer, run_id, run_name = train_adapter(finetune_adapter, train_loader, val_loader, max_epochs=20)

    results = test_adapter(finetune_adapter, trainer, test_loader)
    _logger.debug(f"Test results: {results}")
    trainer.logger.experiment.finish()

    # Additional PD-specific fine-tuning for disease-gene dataset
    pd_trainer = None
    pd_run_id = None
    if conf.finetune.dataset == "disease-gene":
        _logger.info("Finetuning on PD-related genes")
        pd_trainer, pd_run_id, _ = train_PD_adapter(
            base_trainer=trainer,
            model=finetune_adapter,
            embeddings=embeddings,
            kg_edges_split=kg_edges_split,
            run_id=run_id,
            run_name=run_name,
        )

    ckpt_dir = FINETUNING_DIR / conf.paths.checkpoint.checkpoint_path.stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(finetune_adapter.state_dict(), ckpt_dir / f"{run_id}_ft.pt")
    trainer.save_checkpoint(ckpt_dir / f"{run_id}_ft.ckpt")

    if pd_trainer and pd_run_id:
        pd_trainer.save_checkpoint(ckpt_dir / f"{pd_run_id}_ft.ckpt")
