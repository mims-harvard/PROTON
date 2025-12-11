import logging.config
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from src.constants import DEFAULT_CONFIG_PATH

from .models import (
    BatchSizes,
    EdgeSets,
    ExplainerConfig,
    FinetuneConfig,
    FixedParams,
    GraphSampling,
    LoggingConfig,
    MiscFiguresConfig,
    NeuroKGConfig,
    OutputOptions,
    PathsConfig,
    ProtonConfig,
    SplitsConfig,
    SweepConfig,
    TrainingConfig,
    TunableParams,
    WandbConfig,
)


class _YamlConfigSettingsSource(PydanticBaseSettingsSource):
    _config_path: Path | None = None

    @classmethod
    def set_config_path(cls, path: Path):
        cls._config_path = path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str] | None:
        if self._config_path is None:
            return None

        encoding = self.config.get("env_file_encoding")
        file_content = self._read_config_file(encoding)

        field_value = file_content.get(field_name)
        return field_value, field_name

    def _read_config_file(self, encoding: str | None) -> dict[str, Any]:
        if self._config_path is None:
            raise ValueError("Config path not set")

        with open(self._config_path, encoding=encoding) as f:
            return yaml.safe_load(f) or {}

    def __call__(self) -> dict[str, Any]:
        return self._read_config_file(self.config.get("env_file_encoding"))


class Config(BaseSettings):
    """
    Configuration for the application.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
    )

    AZURE_OPENAI_ENDPOINT: str | None = Field(default=None, description="The endpoint for the Azure OpenAI service.")
    AZURE_OPENAI_API_KEY: str | None = Field(default=None, description="The API key for the Azure OpenAI service.")

    seed: int
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig
    neurokg: NeuroKGConfig
    proton: ProtonConfig
    explainer: ExplainerConfig
    wandb: WandbConfig
    splits: SplitsConfig
    finetune: FinetuneConfig
    sweep: SweepConfig
    misc_figures: MiscFiguresConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


def load_config(config_path: Path) -> Config:
    """
    Loads the configuration from the given path.

    Returns:
        Config: The loaded configuration.
    """
    _YamlConfigSettingsSource.set_config_path(config_path)
    return Config()  # ty: ignore[missing-argument]


try:
    conf = load_config(DEFAULT_CONFIG_PATH)
    logging.config.dictConfig(conf.logging.model_dump())
except FileNotFoundError as e:
    raise FileNotFoundError(f"Config file not found at {DEFAULT_CONFIG_PATH}") from e
except Exception as e:
    raise ValueError(f"Failed to load configuration: {e}") from e

__all__ = ["conf", "map_hparams_to_config"]


def map_hparams_to_config(ckpt_hparams: dict) -> ProtonConfig:
    # Map tunable params
    tunable_params = TunableParams(
        num_feat=ckpt_hparams.get("num_feat"),
        num_heads=ckpt_hparams.get("num_heads"),
        hidden_dim=ckpt_hparams.get("hidden_dim"),
        output_dim=ckpt_hparams.get("output_dim"),
        lr=ckpt_hparams.get("lr"),
        wd=ckpt_hparams.get("wd"),
        dropout_prob=ckpt_hparams.get("dropout_prob"),
        max_epochs=ckpt_hparams.get("max_epochs"),
    )

    # Map fixed params
    fixed_params = FixedParams(
        kg="NeuroKG",
        num_layers=ckpt_hparams.get("num_layers"),
        pred_threshold=ckpt_hparams.get("pred_threshold"),
        sampler_fanout=ckpt_hparams.get("sampler_fanout"),
        fixed_k=ckpt_hparams.get("fixed_k"),
        negative_k=ckpt_hparams.get("negative_k"),
        grad_clip=ckpt_hparams.get("grad_clip"),
        lr_factor=ckpt_hparams.get("lr_factor"),
        lr_total_iters=ckpt_hparams.get("lr_total_iters"),
        eps=ckpt_hparams.get("eps"),
    )

    # Map batch sizes
    batch_sizes = BatchSizes(
        train_batch_size=ckpt_hparams.get("train_batch_size"),
        val_batch_size=ckpt_hparams.get("val_batch_size"),
        test_batch_size=ckpt_hparams.get("test_batch_size"),
    )

    # Map graph sampling
    graph_sampling = GraphSampling(
        subsample_graph=ckpt_hparams.get("subsample_graph", False),
        degree_threshold=ckpt_hparams.get("degree_threshold"),
    )

    # Map edge sets
    edge_sets = EdgeSets(
        edge_set=ckpt_hparams.get("edge_set", "hidden"),
    )

    # Map output options
    output_options = OutputOptions(
        save_embeddings=ckpt_hparams.get("save_embeddings", False),
        save_embedding_layer=ckpt_hparams.get("save_embedding_layer", False),
        save_decoder=ckpt_hparams.get("save_decoder", False),
        debug=ckpt_hparams.get("debug", False),
    )

    # Map training config
    training = TrainingConfig(
        batch_sizes=batch_sizes,
        graph_sampling=graph_sampling,
        edge_sets=edge_sets,
        output_options=output_options,
        time_limit=ckpt_hparams.get("time_limit", "02:00:00:00"),
        num_workers=ckpt_hparams.get("num_workers"),
        log_every_n_steps=ckpt_hparams.get("log_every_n_steps"),
        time=ckpt_hparams.get("time", False),
        verbose=ckpt_hparams.get("verbose", True),
        sample_subgraph=ckpt_hparams.get("sample_subgraph", False),
        seed_node=ckpt_hparams.get("seed_node"),
        n_walks=ckpt_hparams.get("n_walks"),
        walk_length=ckpt_hparams.get("walk_length"),
        profiler=ckpt_hparams.get("profiler"),
        resume=ckpt_hparams.get("resume"),
    )

    return ProtonConfig(
        huggingface_repository=ckpt_hparams.get("huggingface_repository", "mims-harvard/PROTON"),
        tunable_params=tunable_params,
        fixed_params=fixed_params,
        training=training,
    )
