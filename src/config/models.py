from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class FinetuneConfig(BaseModel):
    dataset: str = Field(description="String denoting the dataset to finetune on.")
    finetune_ckpt: str | None = Field(
        description="Name of checkpoint for fine-tuned adapter (e.g. '2025_02_02_00_01_28_disease-disease_protein-gene_protein_ft.ckpt')."
    )
    random_baseline: bool = Field(description="Randomly shuffle embeddings as a baseline.")


class SplitsConfig(BaseModel):
    model_name: str = Field(description="Model to use for disease embeddings.")
    batch_size: int = Field(description="Batch size for disease embeddings.")
    embedding_threshold: float = Field(description="Threshold for embedding similarity.")
    levenshtein_threshold: int = Field(description="Threshold for Levenshtein distance.")
    neighborhood_threshold: float = Field(description="Threshold for neighborhood similarity.")
    disease_split: dict[str, int] = Field(description="Disease names and their node indices for the test set.")


class KgPathsConfig(BaseModel):
    base_dir: Path = Field(description="Base directory for knowledge graph data.")
    nodes_path: Path = Field(description="Relative path to node list (converted to full Path during validation).")
    edges_path: Path = Field(description="Relative path to edge list (converted to full Path during validation).")

    @model_validator(mode="after")
    def resolve_paths(self):
        self.nodes_path = self.base_dir / self.nodes_path
        self.edges_path = self.base_dir / self.edges_path
        return self


class CheckpointPathsConfig(BaseModel):
    base_dir: Path = Field(description="Base directory for checkpoints and embeddings.")
    checkpoint_path: Path = Field(
        description="Relative path to checkpoint file (converted to full Path during validation)."
    )
    embeddings_path: Path = Field(
        description="Relative path to embeddings file (converted to full Path during validation)."
    )

    @model_validator(mode="after")
    def resolve_paths(self):
        self.checkpoint_path = self.base_dir / self.checkpoint_path
        self.embeddings_path = self.base_dir / self.embeddings_path
        return self


class MappingsPathsConfig(BaseModel):
    base_dir: Path = Field(description="Base directory for mappings.")
    hgnc_path: Path = Field(description="Relative path to the HGNC file (converted to full Path during validation).")
    drug_path: Path = Field(
        description="Relative path to the drug mappings file (converted to full Path during validation)."
    )
    mondo_efo_path: Path = Field(
        description="Relative path to the MONDO EFO mappings file (converted to full Path during validation)."
    )

    @model_validator(mode="after")
    def resolve_paths(self):
        self.hgnc_path = self.base_dir / self.hgnc_path
        self.drug_path = self.base_dir / self.drug_path
        self.mondo_efo_path = self.base_dir / self.mondo_efo_path
        return self


class NotebooksPathsConfig(BaseModel):
    base_dir: Path = Field(description="Base directory for notebook-related files.")
    asyn_screens: Path = Field(
        description="Relative path to the wet lab alpha-synuclein screen results (converted to full Path during validation)."
    )
    private_ehr_dir: Path = Field(
        description="Directory for EHR validation data (empty string means external/full path)."
    )
    pd_related_genes: Path = Field(
        description="Relative path to the gene list (converted to full Path during validation)."
    )
    gwas_catalog: Path = Field(
        description="Relative path to the GWAS catalog (converted to full Path during validation)."
    )
    pesticide_lists: Path = Field(
        description="Relative path to the pesticide lists (converted to full Path during validation)."
    )
    drug_repurposing_dir: Path = Field(
        description="Relative path to drug repurposing outputs (converted to full Path during validation)."
    )
    cohort_analysis_dir: Path = Field(
        description="Relative path to EHR cohort analysis outputs (converted to full Path during validation)."
    )
    asyn_screens_dir: Path = Field(
        description="Relative path to in silico alpha synuclein screens outputs (converted to full Path during validation)."
    )
    kg_viz_dir: Path = Field(
        description="Relative path to miscellaneous figures (converted to full Path during validation)."
    )
    essentiality_gwas_dir: Path = Field(
        description="Relative path to multi-disease essentiality outputs (converted to full Path during validation)."
    )
    organoid_validation_dir: Path = Field(
        description="Relative path to organoid validation outputs (converted to full Path during validation)."
    )
    pesticide_prediction_dir: Path = Field(
        description="Relative path to pesticide prediction outputs (converted to full Path during validation)."
    )

    @model_validator(mode="after")
    def resolve_paths(self):
        """Convert relative paths to full Path objects, handling empty strings as external paths."""
        # Paths that should be joined with base_dir
        path_fields = [
            "asyn_screens",
            "pd_related_genes",
            "gwas_catalog",
            "pesticide_lists",
            "drug_repurposing_dir",
            "cohort_analysis_dir",
            "asyn_screens_dir",
            "kg_viz_dir",
            "essentiality_gwas_dir",
            "organoid_validation_dir",
            "pesticide_prediction_dir",
        ]

        for field_name in path_fields:
            rel_path = getattr(self, field_name)
            setattr(self, field_name, self.base_dir / rel_path)

        return self


class PathsConfig(BaseModel):
    kg: KgPathsConfig = Field(description="Knowledge graph paths configuration.")
    checkpoint: CheckpointPathsConfig = Field(description="Checkpoint and embeddings paths configuration.")
    mappings: MappingsPathsConfig = Field(description="Mappings paths configuration.")
    notebooks: NotebooksPathsConfig = Field(description="Notebooks paths configuration.")
    embeddings_cache_path: Path = Field(description="Path to cache disease embeddings.")
    splits_dir: Path = Field(description="Path to splits directory.")
    split_emb_dir: Path = Field(description="Path to split embeddings directory.")
    secrets_path: Path = Field(description="Path to the secrets file.")
    mondo_obo_path: Path = Field(description="Path to the MONDO OBO file.")
    pqa_prompts_dir: Path = Field(description="Directory for saving PQA prompts.")
    pqa_results_dir: Path = Field(description="Directory for saving PQA results.")
    sweep_output_dir: Path = Field(description="Path to sweep output directory.")
    wandb_save_dir: Path = Field(description="Directory for saving wandb files.")
    explainer_dir: Path = Field(description="Directory for explainer outputs.")


class NeuroKGConfig(BaseModel):
    dataverse_base_url: str = Field(description="Base URL for the Dataverse API.")
    dataverse_persistent_id: str = Field(description="Persistent ID for the NeuroKG dataset.")
    full_kg: bool = Field(description="Use full KG for training, no validation or test set.")
    test_set: str | None = Field(
        None,
        description=r"Index of the disease split to use as the test set, e.g. '42049'. If None, no test set sampled randomly.",
    )


class TunableParams(BaseModel):
    num_feat: int = Field(description="Dimension of embedding layer.")
    num_heads: int
    hidden_dim: int
    output_dim: int
    lr: float
    wd: float
    dropout_prob: float = Field(description="Dropout probability.")
    max_epochs: int


class FixedParams(BaseModel):
    kg: str
    num_layers: int
    pred_threshold: float
    sampler_fanout: list[int]
    fixed_k: int
    negative_k: int
    grad_clip: float
    lr_factor: float
    lr_total_iters: int
    eps: float


class BatchSizes(BaseModel):
    train_batch_size: int = Field(description="Training batch size.")
    val_batch_size: int = Field(description="Validation batch size.")
    test_batch_size: int = Field(description="Test batch size.")


class GraphSampling(BaseModel):
    subsample_graph: bool
    degree_threshold: int


class EdgeSets(BaseModel):
    edge_set: str = Field(description="Edge set to score. Options: hidden, training, etc.")


class OutputOptions(BaseModel):
    save_embeddings: bool
    save_embedding_layer: bool
    save_decoder: bool
    debug: bool


class TrainingConfig(BaseModel):
    batch_sizes: BatchSizes
    graph_sampling: GraphSampling
    edge_sets: EdgeSets
    output_options: OutputOptions
    time_limit: str = Field(
        description="Time limit for training in DD:HH:MM:SS format.",
        pattern=r"^\d{2}:\d{2}:\d{2}:\d{2}$",
        examples=["02:00:00:00"],
    )
    num_workers: int
    log_every_n_steps: int
    time: bool
    verbose: bool
    sample_subgraph: bool
    seed_node: int
    n_walks: int
    walk_length: int
    profiler: str | None
    resume: str | None = Field(
        None,
        description="Run ID to resume training from (format: YYYY-MM-DD_UUID).",
    )


class ProtonConfig(BaseModel):
    huggingface_repository: str = Field(description="Hugging Face repository.")
    tunable_params: TunableParams
    fixed_params: FixedParams
    training: TrainingConfig


class WandbConfig(BaseModel):
    pretrain_project_name: str = Field(description="Project name for pretraining.")
    splits_project_name: str = Field(description="Project name for disease splits.")
    finetune_project_name: str = Field(description="Project name for finetuning.")
    entity_name: str = Field(description="Entity name for Weights and Biases.")


class ExplainerConfig(BaseModel):
    lr: float
    num_epochs: int
    sparsity_loss_alpha: float
    entropy_loss_alpha: float
    khop: int
    degree_threshold: int
    query_edge_types: list[tuple[str, str, str]]
    src_indices: list[int]
    dst_indices: list[int]

    @field_validator("query_edge_types")
    def validate_query_edge_types(cls, v):
        return [tuple(edge_type) for edge_type in v]


class SweepConfig(BaseModel):
    wandb_path: str = Field(description="Path to Weights and Biases sweep.")
    wandb_runs_path: str = Field(description="Path to Weights and Biases runs.")


class MiscFiguresConfig(BaseModel):
    seed_indices: dict[str, int] = Field(description="Node indices for diseases.")


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool
    formatters: dict[str, Any]
    handlers: dict[str, Any]
    loggers: dict[str, Any]
    root: dict[str, Any]
