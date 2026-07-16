# BINN / NicheTrans Agent README

This repository is a notebook-first PyTorch project for NicheTrans spatial multi-omics translation experiments. It predicts target modalities such as MSI metabolites, proteins, pathology labels, RNA, or ATAC from a source omics panel plus spatial neighbors, with optional image, cell-type, and static gene-prior signals.

This README is written for coding agents. Prefer it as the project map before editing code or notebooks.

## Quick Start

Run commands from the repository root (`D:\BINN` in the current workspace) so imports like `from model.nicheTrans import NicheTrans` resolve without packaging the project. Prefer the repository-local GPU Python instead of the system `python`:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$py = ".\.conda\gene-prior-gpu\python.exe"
& $py -m pytest prior_AddOn/tests
.\.conda\gene-prior-gpu\Scripts\jupyter.exe lab
```

Notes:

- There is no `setup.py` or `pyproject.toml`; this is not an installable package yet.
- Most default dataset paths in `args/` point to the original author's Linux filesystem. Override them in notebooks or CLI arguments for local runs.
- `requirements.txt` contains CUDA/PyTorch-specific pins. Treat it as a dependency reference for rebuilding or filling missing packages, not as a command to blindly overwrite the repository GPU environment.
- Local environment and large artifact folders such as `.conda/`, `.deps/`, `.cache/`, and generated prior embedding binaries are intentionally ignored.

## Repository GPU Environment

Use `.conda/gene-prior-gpu` for GPU work in this repository.

| Requirement | Repository-local value / expectation |
| --- | --- |
| Python executable | `.\.conda\gene-prior-gpu\python.exe` |
| Python version | 3.12.9 in the current workspace |
| PyTorch stack | `torch 2.7.0+cu126`, `torchvision 0.22.0+cu126` in the current workspace |
| CUDA runtime seen by PyTorch | 12.6 (`torch.version.cuda`) |
| GPU availability | `torch.cuda.is_available()` should be `True` for training runs |
| Notebook launcher | `.\.conda\gene-prior-gpu\Scripts\jupyter.exe lab` |
| Project imports | Run from repo root or set `PYTHONPATH` to the repo root |

Verify the environment before training:

```powershell
@'
import sys
import torch
print(sys.executable)
print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
'@ | .\.conda\gene-prior-gpu\python.exe -
```

If the GPU check is `False`, stop before long training runs and fix the NVIDIA driver / CUDA-compatible PyTorch environment first. Dataset notebooks also require the scientific stack used in `requirements.txt` (`scanpy`, `anndata`, `scikit-misc`, `pandas`, `scikit-learn`, `opencv-python`, `captum`, etc.); install missing packages into `.conda/gene-prior-gpu` only after preserving a compatible CUDA PyTorch build.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Tutorial_*.ipynb` | Canonical experiment entry points. Training, visualization, and attribution workflows live here. |
| `args/` | Per-dataset `argparse` defaults for paths, training hyperparameters, batch sizes, and GPU ids. |
| `datasets/` | Eager data managers and PyTorch `Dataset` wrappers. Managers build spatial graphs and materialize train/test lists in memory. |
| `model/` | NicheTrans model variants, self-attention blocks, attribution variants, and static prior pooling/fusion modules. |
| `utils/` | DataLoader builders, train/test loops, seed helpers, and metrics. |
| `prior_AddOn/` | Static scGPT/Geneformer gene-prior loading, filtering, ablation utilities, cache/report scripts, and unit tests. |
| `NicheTrans_STARmap_PLUS_baseline.pth` | Baseline checkpoint artifact kept at repo root. Do not overwrite it unless explicitly asked. |

## Tutorial Index

| Tutorial | Dataset/workflow | Main code path |
| --- | --- | --- |
| `Tutorial_3.*` | SMA RNA to MSI, visualization, attribution | `datasets.data_manager_SMA`, `model.nicheTrans_img`, `utils_training_SMA` |
| `Tutorial_4.*` | NicheTrans* on SMA | `model.nicheTrans_img` |
| `Tutorial_5.*` | STARmap PLUS AD mouse, static gene-prior capable | `datasets.data_manager_STARmap_PLUS`, `model.nicheTrans`, `prior_AddOn` |
| `Tutorial_6.*` | NicheTrans* STARmap PLUS with cell-type info and attribution | `model.nicheTrans_ct`, `model.nicheTrans_ct_attribution_STARmap_PLUS` |
| `Tutorial_7.*` | 10x Xenium breast cancer, static gene-prior capable | `datasets.data_manager_breast_cancer`, `model.nicheTrans`, `prior_AddOn` |
| `Tutorial_8.*` | Human lymph node RNA to protein | `datasets.data_manager_human_lymph_node`, `model.nicheTrans_img` |
| `Tutorial_9.*` | MISAR-seq ATAC to RNA / RNA to ATAC | `datasets.data_manager_MISAR_seq`, `model.nicheTrans_hd` |
| `Tutorial_10.1` | Schema-compliant H5MU baseline/static-prior training and comparison | `datasets.h5mu_dataset`, `model.nicheTrans`, `prior_AddOn`, `utils.utils_training_h5mu` |

`Tutorial_7.1__Train_NicheTrans_on_10x_Xenium_data copy.ipynb` is a duplicate/older copy. Prefer `Tutorial_7.1__Train_NicheTrans_on_10x_Xenium_data.ipynb` unless the user asks about the copy.

## Core Data Flow

1. A notebook imports an `args/*.py` generator and a dataset manager from `datasets/`.
2. The dataset manager reads `.h5ad`, image, CSV, Excel, or Visium inputs, normalizes/features-selects data, builds spatial neighbors, and stores prepared samples in `dataset.training`, `dataset.testing`, and sometimes `dataset.val`.
3. `utils/utils_dataloader.py` wraps those lists with PyTorch `Dataset` classes from `datasets/data_loader.py`.
4. A model from `model/` consumes source expression and source-neighbor tensors. Some variants also consume images or cell-type tensors.
5. A dataset-specific training loop in `utils/utils_training_*.py` handles augmentation, loss, evaluation, and metric printing.

Important shape convention:

- Center source tensor: `[batch, source_length]`
- Neighbor source tensor: `[batch, n_neighbors, source_length]`
- Target tensor: `[batch, target_length]`
- `dataset.source_panel` order must match source tensor columns.
- `dataset.target_panel` order must match target tensor columns.

### Schema-compliant H5MU files

Use `H5MuDataManager` when training and testing data are supplied as two
independent spatial multi-omics `.h5mu` files. The manager validates both files,
requires identical source/target feature names and order, and builds spatial
neighbors independently within each file and `sample_id`.

```python
from datasets.h5mu_dataset import H5MuDataManager, validate_h5mu
from utils.utils_h5mu_dataloader import h5mu_dataloader

validate_h5mu("train.h5mu")
dataset = H5MuDataManager(
    train_path="train.h5mu",
    test_path="test.h5mu",
    source_modality="rna",
    target_modality="protein",
    n_neighbors=12,
    preprocess="auto",
)
trainloader, testloader = h5mu_dataloader(args, dataset)
```

Each batch contains `(source, target, source_neighbors, obs_id)`. Automatic
preprocessing normalizes count rows to a total of `1e3` before `log1p`, applies
`log1p` to non-negative intensity data, and converts all model inputs to
`float32`. Pass `preprocess="raw"` to disable these numeric transformations.
Backed HDF5 handles are opened lazily per DataLoader worker, so the complete
matrices are not densified in memory.

`Tutorial_10.1__Train_NicheTrans_on_H5MU_data.ipynb` provides the corresponding
standard NicheTrans workflow. It treats the testing H5MU as a periodic validation
set, selects the best checkpoint by mean Pearson correlation (regression) or mean
AUROC (binary), and supports three direct notebook modes:

- `baseline`: train without static priors on the complete source panel. This mode
  does not load or require prior artifacts.
- `prior`: align and filter a selected scGPT/Geneformer prior, then train with QKV
  prior pooling.
- `compare`: train baseline and prior variants on the same prior-covered source
  panel and seed, then report prior-minus-baseline metric deltas.

Configure these switches in the notebook rather than through `args_h5mu.py`:

```python
EXPERIMENT_MODE = "compare"  # baseline | prior | compare
PRIOR_MODEL = "scgpt"  # scgpt | geneformer
PRIOR_ROOT = Path("prior_AddOn/gene_embeddings")
PRIOR_ALLOW_NETWORK = False
PRIOR_NORMALIZE_EMBEDDING = True

PARALLEL_COMPARE = True
PARALLEL_GPU_IDS = (0, 0)  # share GPU 0; use (0, 1) for two GPUs
```

Prior-enabled modes require RNA as the source modality. The notebook reads and
canonicalizes human/mouse species from `uns['database']['organism']`, requires
matching train/test organisms, aligns priors to `dataset.source_panel`, and filters
the source panel before creating any DataLoader. In `compare` mode the baseline
uses that same filtered panel so the comparison does not confound prior fusion
with a different number of input genes.

With `PARALLEL_COMPARE=True`, joblib's `loky` backend launches baseline and prior
as two independent processes. Each process opens its own backed H5MU handles and
builds its own model, optimizer, scheduler, and DataLoaders. Repeating a GPU id,
for example `(0, 0)`, runs both processes concurrently on one GPU; this requires
enough memory for both training jobs and may be slower if the GPU is saturated.
Use `(0, 1)` for two visible GPUs or set `PARALLEL_COMPARE=False` for sequential
training. With the default `device="auto"`, missing/invisible configured CUDA
devices trigger a sequential CPU/GPU fallback; an explicit unavailable
`device="cuda"` request still raises an error. Parallel workers do not wrap their
models in `DataParallel`. They also force each inner PyTorch DataLoader to
`workers=0`: spawning DataLoader worker processes from inside a `loky` process is
unsupported and can fail with `AttributeError: 'Process' object has no attribute
'env'`. The configured `--workers` value remains active for sequential runs.

Each variant writes a distinct best checkpoint, history CSV, and per-feature
validation CSV. The final `{run_name}_{mode}_comparison.csv` contains both summary
rows and, when both variants are present, `*_delta_vs_baseline` columns. Positive
deltas are better for Pearson/Spearman/AUROC; negative deltas are better for loss
and RMSE.

This path requires `torch`, `mudata`, `anndata`, `h5py`, `numpy`, `pandas`,
`scipy`, and `joblib`. Binary evaluation additionally requires `scikit-learn`.
It does not require Scanpy, MuON, torchvision, or Pillow. Install `pytest`
separately to run its automated tests.

## Model Variants

- `model/nicheTrans.py`: spatial omics-only NicheTrans. This is the variant extended with static gene-prior support.
- `model/nicheTrans_img.py`: image + omics NicheTrans using ResNet18 features.
- `model/nicheTrans_ct.py`: NicheTrans* variant with cell-type tokens.
- `model/nicheTrans_hd.py`: high-dimensional output variant with one shared prediction head.
- `model/*attribution*.py`: attribution-specific variants used by attribution notebooks.
- `model/gene_prior_pooling.py`: QKV static gene-prior pooling and gated fusion used by `model/nicheTrans.py`.

## Static Gene Prior Workflow

The static prior path aligns scGPT/Geneformer gene embeddings to `dataset.source_panel` and can inject them into `NicheTrans`.

Build/export static embeddings:

```powershell
python prior_AddOn/build_static_gene_embeddings.py --root prior_AddOn/gene_embeddings --models scgpt geneformer
```

Load and filter before creating dataloaders or the model:

```python
from prior_AddOn.gene_embedding_loader import load_static_gene_prior
from prior_AddOn.gene_prior_filter import filter_dataset_by_gene_prior
from model.nicheTrans import NicheTrans

prior_model = "scgpt"
priors = load_static_gene_prior(
    source_panel=dataset.source_panel,
    species="mouse",  # or "human"
    models=(prior_model,),
    root="prior_AddOn/gene_embeddings",
)
dataset, priors, filter_info = filter_dataset_by_gene_prior(
    dataset=dataset,
    priors=priors,
    prior_model=prior_model,
)
model = NicheTrans(
    source_length=dataset.rna_length,
    target_length=dataset.target_length,
    priors=priors,
    prior_model=prior_model,
    prior_pooling_mode="qkv",
)
```

For H5MU experiments, Tutorial 10.1 performs this workflow automatically. It
uses the H5MU organism metadata instead of a manually entered species and passes
the resulting `keep_mask` to every independently constructed experiment dataset.

Agent rules for prior work:

- `filter_dataset_by_gene_prior` mutates and returns the dataset. Create dataloaders only after filtering.
- `NicheTrans(..., prior_pooling_mode="qkv")` requires filtered priors with no missing genes in `found_mask`; otherwise construction fails.
- When multiple prior models are passed, always pass `prior_model` explicitly.
- `randomize_static_gene_prior` returns a copied prior dict for ablation and uses a private CPU RNG so it does not advance global training RNG state.
- For mouse panels, ortholog mapping may call Ensembl unless `allow_network=False`; cached mappings live under `prior_AddOn/gene_embeddings/processed/`.
- See `prior_AddOn/LOAD_STATIC_GENE_PRIOR.md` for detailed cache semantics and offline mode.

Useful prior commands:

```powershell
python prior_AddOn/build_starmap_plus_mouse_mapping.py --embedding-root prior_AddOn/gene_embeddings --ad-adata-path <STARmap_PLUS_AD_h5ad_dir>
python -m pytest prior_AddOn/tests
```

## Testing And Verification

Primary automated tests currently cover the `prior_AddOn` modules:

```powershell
python -m pytest prior_AddOn/tests
```

For model or data-manager changes, unit tests may not be enough because many paths require real `.h5ad` data. When data is unavailable, do at least:

```powershell
python -m compileall args datasets model utils prior_AddOn
```

Run all H5MU/prior regression tests in the `iscdc` environment with:

```powershell
$py = "C:\Users\shetao\.conda\envs\iscdc\python.exe"
& $py -m pytest datasets/tests/test_h5mu_dataset.py utils/tests/test_utils_training_h5mu.py prior_AddOn/tests
```

For Tutorial 10.1 changes, validate the notebook JSON/code-cell syntax and run
small sequential smoke experiments with `PARALLEL_COMPARE=False` for `baseline`,
`prior`, and `compare`. GPU-process concurrency should be tested only in a CUDA
environment with sufficient memory; do not infer GPU readiness from CPU smoke
tests.

## Editing Guidance For Agents

- Preserve notebook-driven workflows. If you change a model or dataset signature, update the corresponding tutorial imports/calls in the same change.
- Do not instantiate dataset managers in tests or smoke checks unless the required local data exists; constructors eagerly read files and preprocess full datasets.
- Keep source-panel order stable. Any gene filtering must filter source arrays, neighbor arrays, `source_panel`, length attributes, and prior embeddings together.
- Avoid overwriting `.pth` checkpoints, generated `.pt` prior artifacts, mapping caches, or notebook outputs unless the user explicitly wants artifact regeneration.
- Be careful with `argparse` modules in notebooks: they call `parser.parse_args()`, so external notebook/kernel arguments can interfere.
- `model/nicheTrans_img.py` uses `torchvision.models.resnet18(pretrained=True)`, which may try to download weights in a fresh environment.
- Existing code uses wildcard imports and simple script-style modules. Prefer small, local changes that match the current style over broad package refactors.
- Some strings and variable names include non-ASCII biology labels. Preserve them unless the user asks for cleanup.
- The repository has a GPL-3.0 `LICENSE`; keep derivative code compatible.

## Common Pitfalls

- Running a training notebook without overriding `args` paths will usually fail on missing original-author data directories.
- Creating dataloaders before gene-prior filtering leads to stale tensor dimensions.
- Passing unfiltered priors to `NicheTrans` with `qkv` pooling raises a shape or missing-gene error by design.
- Running both compare workers on one GPU can exhaust memory because each process owns a complete model, optimizer, batch, and CUDA allocator. Reduce batch sizes or use sequential execution when needed.
- Parallel compare intentionally uses `dataloader_workers=0` inside each `loky` process to avoid nested multiprocessing; use the recorded column to confirm the effective value.
- With the default `device="auto"`, `PARALLEL_COMPARE=True` falls back to sequential execution when CUDA or a configured GPU id is unavailable; inspect the warning and the recorded `process_id`, `device`, and `assigned_gpu_id` columns when confirming execution mode.
- Changing the number/order of neighbors can break spatial token assumptions in some model variants. `nicheTrans.py` computes tokens from neighbor length; older variants hard-code repeat counts.
- `requirements.txt` may not be portable across OS/CUDA combinations because of binary package pins.
