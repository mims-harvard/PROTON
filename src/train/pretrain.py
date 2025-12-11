import logging
import uuid
from datetime import datetime

import dgl
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from pytorch_lightning.loggers import WandbLogger

from src.config import conf
from src.dataloaders import create_dataloaders, load_graph, partition_graph
from src.models import HGT

_logger = logging.getLogger(__name__)


def _setup_graph_and_dataloaders() -> tuple[
    dgl.DGLGraph,
    dgl.dataloading.DataLoader | None,
    dgl.dataloading.DataLoader | None,
    dgl.dataloading.DataLoader | None,
]:
    """Load graph, partition, and create dataloaders."""
    kg = load_graph()
    if conf.neurokg.full_kg:
        train_kg, val_kg, test_kg = kg, None, None
    else:
        train_kg, val_kg, test_kg = partition_graph(kg)

    train_dl, val_dl, test_dl = create_dataloaders(
        kg,
        train_kg,
        val_kg,  # ty: ignore[invalid-argument-type]
        test_kg,  # ty: ignore[invalid-argument-type]
        sampler_fanout=conf.proton.fixed_params.sampler_fanout,
        fixed_k=conf.proton.fixed_params.fixed_k,
        negative_k=conf.proton.fixed_params.negative_k,
        train_batch_size=conf.proton.training.batch_sizes.train_batch_size,
        val_batch_size=conf.proton.training.batch_sizes.val_batch_size,
        test_batch_size=conf.proton.training.batch_sizes.test_batch_size,
        num_workers=conf.proton.training.num_workers,
        subsample_graph=conf.proton.training.graph_sampling.subsample_graph,
        full_kg=conf.neurokg.full_kg,
    )
    return kg, train_dl, val_dl, test_dl


def _setup_run(kg: dgl.DGLGraph) -> tuple[HGT, WandbLogger, str]:
    """Setup a new or resumed training run."""
    if conf.proton.training.resume:
        run_id = conf.proton.training.resume
        if not conf.paths.checkpoint.checkpoint_path:
            raise ValueError("Best checkpoint must be specified for resuming.")
        if run_id != conf.paths.checkpoint.checkpoint_path.name.split("_epoch")[0]:
            raise ValueError("Run ID from checkpoint does not match provided run ID.")

        run_id_time, run_uuid = run_id.split("_")
        run_time = datetime.strptime(run_id_time, "%Y-%m-%d")
        run_name = f"{run_uuid}{run_time.strftime(' on %m/%d/%Y')}"
        resume_status = "must"
        model = HGT.load_from_checkpoint(
            checkpoint_path=str(conf.paths.checkpoint.checkpoint_path),
            kg=kg,
            hparams=conf.neurokg,
            strict=False,
        )
    else:
        curr_time = datetime.now()
        run_uuid = str(uuid.uuid4())[:8]
        run_name = f"{run_uuid}{curr_time.strftime(' on %m/%d/%Y')}"
        run_id = f"{curr_time.strftime('%Y-%m-%d')}_{run_uuid}"
        resume_status = "allow"
        model = HGT(kg=kg, hparams=conf.neurokg)

    project_name = conf.wandb.splits_project_name if conf.neurokg.test_set else conf.wandb.pretrain_project_name
    if conf.neurokg.test_set:
        run_name += f" ({conf.neurokg.test_set})"

    wandb_logger = WandbLogger(
        name=run_name, project=project_name, save_dir=conf.paths.wandb_save_dir, id=run_id, resume=resume_status
    )
    return model, wandb_logger, run_id


def _setup_callbacks(run_id: str, val_dataloader_exists: bool) -> list[pl.Callback]:
    """Setup PyTorch Lightning callbacks."""
    checkpoint_filename = f"{run_id}_{{epoch}}-{{step}}"
    if conf.neurokg.test_set:
        checkpoint_filename += f"-test={conf.neurokg.test_set}"

    dirpath = conf.paths.checkpoint.base_dir

    if val_dataloader_exists:
        monitor = "val/auroc"
        early_stopping = EarlyStopping(monitor=monitor, patience=4, mode="max", verbose=True)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=dirpath,
            filename=checkpoint_filename,
            save_top_k=1,
            mode="max",
            verbose=1,  # ty: ignore[invalid-argument-type]
        )
        callbacks = [checkpoint_callback, early_stopping]
    else:
        monitor = "train/auroc"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=dirpath,
            filename=checkpoint_filename,
            save_top_k=1,
            mode="max",
            verbose=1,  # ty: ignore[invalid-argument-type]
        )
        callbacks = [checkpoint_callback]

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(Timer(duration=conf.proton.training.time_limit))
    return callbacks


def _get_trainer(wandb_logger: WandbLogger, callbacks: list[pl.Callback], val_dataloader_exists: bool) -> pl.Trainer:
    """Configure and return the PyTorch Lightning trainer."""
    if conf.proton.training.output_options.debug:
        limit_train_batches, limit_val_batches = 5, 1
        conf.proton.tunable_params.max_epochs = 3
        conf.proton.training.log_every_n_steps = 1  # ty: ignore[invalid-assignment]
    else:
        limit_train_batches, limit_val_batches = 1.0, 1.0

    return pl.Trainer(
        devices=1 if torch.cuda.is_available() else 0,
        accelerator="gpu",
        logger=wandb_logger,
        max_epochs=conf.proton.tunable_params.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=conf.proton.fixed_params.grad_clip,
        profiler=conf.proton.training.profiler,
        log_every_n_steps=conf.proton.training.log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_check_interval=0.25 if val_dataloader_exists else 1.0,
        deterministic=True,
    )


def pretrain():
    pl.seed_everything(conf.seed, workers=True)

    kg, train_dataloader, val_dataloader, test_dataloader = _setup_graph_and_dataloaders()

    model, wandb_logger, run_id = _setup_run(kg)

    model_params = sum(p.numel() for p in model.parameters())
    emb_params = model.emb.weight.numel()
    _logger.info(f"Total model parameters: {model_params}")
    _logger.info(f"Embedding layer parameters: {emb_params}")
    _logger.info(f"Trainable model parameters: {model_params - emb_params}")

    wandb_logger.watch(model, log="all")

    callbacks = _setup_callbacks(run_id, val_dataloader is not None)
    trainer = _get_trainer(wandb_logger, callbacks, val_dataloader is not None)

    if val_dataloader:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader)

    if test_dataloader:
        trainer.test(model, test_dataloader)
