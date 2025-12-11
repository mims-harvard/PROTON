<!-- PROTON: Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems -->
<p align="center">
<img src="data/images/header.png?raw=true" width="100%" title="PROTON: Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems">
</p>

[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge)](https://protonmodel.ai)
[![Paper](https://img.shields.io/badge/Biorxiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org)
[![Code](https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mims-harvard/PROTON)
[![Model](https://img.shields.io/badge/Model-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ayushnoori/PROTON)


## Introduction

Neurological diseases are the leading global cause of disability, yet most lack disease-modifying treatments. We present PROTON, a heterogeneous graph transformer that generates testable hypotheses across molecular, organoid, and clinical systems. To evaluate PROTON, we apply it to Parkinson's disease (PD), bipolar disorder (BD), and Alzheimer's disease (AD). In PD, PROTON linked genetic risk loci to genes essential for dopaminergic neuron survival and predicted pesticides toxic to patient-derived neurons, including the insecticide endosulfan, which ranked within the top 1.29\% of predictions. *In silico* PROTON screens reproduced six genome-wide $\alpha$-synuclein experiments, including a split-ubiquitin yeast two-hybrid system (normalized enrichment score [NES] = 2.30, FDR-adjusted $p < 1 \times 10^{-4}$), an ascorbate peroxidase proximity labeling assay (NES = 2.16, FDR $< 1 \times 10^{-4}$), and a high-depth targeted exome sequencing study in 496 synucleinopathy patients (NES = 2.13, FDR $< 1 \times 10^{-4}$). In BD, PROTON predicted calcitriol as a candidate drug that reversed proteomic alterations observed in cortical organoids derived from BD patients. In AD, we evaluated PROTON predictions in health records from $n$ = 610,524 patients at Mass General Brigham, confirming that five PROTON-predicted drugs were associated with reduced seven-year dementia risk (minimum hazard ratio = 0.63, 95% CI: 0.53–0.75, $p < 1 \times 10^{-7}$). PROTON generated neurological hypotheses that were evaluated across molecular, organoid, and clinical systems, defining a path for AI-driven discovery in neurological disease.

<p align="center">
<img src="data/images/figure_1a.png?raw=true" width="100%" title="PROTON is a graph AI model for neurological disease.">
</p>

## Training Data

PROTON was trained on NeuroKG, a heterogeneous, undirected biomedical knowledge graph contextualized to the human brain. NeuroKG unifies 36 human datasets and ontologies, and integrates single-nucleus RNA-sequencing atlases comprising 3,756,702 cells from the adult human brain. The knowledge graph contains 147,020 nodes across 16 entity types and 7,366,745 edges across 47 relation types. NeuroKG is available via Harvard Dataverse at DOI: [10.7910/DVN/ZDLS3K](https://doi.org/10.7910/DVN/ZDLS3K). For more details, please refer to our [project website](https://protonmodel.ai).

## Model Architecture

PROTON is a 578-million-parameter heterogeneous graph transformer for neurological disease. It was trained on NeuroKG using a self-supervised link prediction objective. Through Bayesian hyperparameter optimization, we selected a model architecture that achieved high link-prediction performance (AUROC = 0.9145; accuracy = 82.23%) on an independent test set. For more details, please refer to our [project website](https://protonmodel.ai).


## Usage Instructions

To use PROTON, please complete the following steps.

1️⃣ First, clone this repository and set up your environment.
```bash
git clone https://github.com/mims-harvard/PROTON.git
cd PROTON
```

Set up and verify your development environment. Note, [GNU Make](https://www.gnu.org/software/make/) and [uv](https://docs.astral.sh/uv/getting-started/installation/) are minimally required.
```bash
make install
make check
```

2️⃣ Download the knowledge graph from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZDLS3K).
```
uv run cli download-neurokg
```

3️⃣ Download the model weights from [Hugging Face](https://huggingface.co/mims-harvard/PROTON).
```
uv run cli download-proton
```

By default, the `disease_splits` folder is excluded from the download. To include it (required for `notebooks/drug_repurposing.ipynb`), use the `--download_splits` flag:
```
uv run cli download-proton --download_splits
```

4️⃣ Modify the `conf/default.config.yaml` file appropriately. For example, if you want to use Weights and Biases for logging, you can set the following:
```yaml
wandb:
  pretrain_project_name: "proton-pretraining"
  entity_name: "ayushnoori"
```

5️⃣ Finally, load the model with the following code:
```python
import pytorch_lightning as pl
from src.config import conf
from src.constants import TORCH_DEVICE
from src.dataloaders import load_graph
from src.models import HGT

pl.seed_everything(conf.seed, workers=True)
kg = load_graph(nodes, edges)
pretrain_model = HGT.load_from_checkpoint(
    checkpoint_path=str(conf.hgt.checkpoint_path),
    kg=kg,
    strict=False,
)
pretrain_model.eval()
pretrain_model = pretrain_model.to(TORCH_DEVICE)
```

To run Jupyter notebooks in `/notebooks` used to create the figures in the paper, you can start a Jupyter server.
```bash
make jupyterlab
```

To see additional available commands, run:
```bash
uv run cli --help
```

For example, to run the random walk with restart (RWR) baseline for PD-related *in silico* screens, use:
```bash
uv run cli random-walks data/neurokg/nodes.csv data/neurokg/edges.csv --start 39579 --length 10 --num-walks 10000

```

Convenient Makefile commands are also included for common development tasks. To see the available commands, run:
```bash
make help
```

## License

PROTON is released under the [MIT License](https://github.com/mims-harvard/PROTON/blob/main/LICENSE).


## Citation

If you use PROTON, please consider citing our paper.
```
@article{noori_graph_2025,
  title={Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems},
  author={Noori, Ayush and Polonuer, Joaquin and Meyer, Katharina and Budnik, Bogdan and Morton, Shad and Wang, Xinyuan and Nazeem, Sumaiya and He, Yingnan and Arango, Iñaki and Vittor, Lucas and Woodworth, Matthew and Krolewski, Richard C. and Li, Michelle M. and Liu, Ninning and Kamath, Tushar and Macosko, Evan and Ritter, Dylan and Afroz, Jalwa and Henderson, Alexander B. H. and Studer, Lorenz and Rodriques, Samuel G. and White, Andrew and Dagan, Noa and Clifton, David A. and Church, George M. and Das, Sudeshna and Tam, Jenny M. and Khurana, Vikram and Zitnik, Marinka},
  journal={arXiv preprint},
  note={arXiv:XXXX.XXXXX (placeholder)},
  year={2025}
}
```


## Contact

For any questions or feedback, please open an issue in the [GitHub repository](https://github.com/mims-harvard/PROTON/issues/new) or contact [Ayush Noori](mailto:ayush.noori@sjc.ox.ac.uk) and [Marinka Zitnik](mailto:marinka@hms.harvard.edu).
