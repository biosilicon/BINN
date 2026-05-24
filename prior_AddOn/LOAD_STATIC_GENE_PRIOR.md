# `load_static_gene_prior` Usage

`load_static_gene_prior` aligns exported static gene embeddings from scGPT and Geneformer to the exact order of an input gene panel. It does not modify the NicheTrans model; it only loads prior embeddings, resolves gene names, handles mouse-to-human ortholog mapping, and returns tensors ready for later experiments.

## Build Static Embeddings First

Run this once before loading priors:

```powershell
D:\BINN\.conda\gene-prior-gpu\python.exe prior_AddOn\build_static_gene_embeddings.py --root D:\BINN\prior_AddOn\gene_embeddings --models scgpt geneformer
```

Expected outputs:

- `D:\BINN\prior_AddOn\gene_embeddings\processed\scgpt_static.pt`
- `D:\BINN\prior_AddOn\gene_embeddings\processed\geneformer_v2_316m_static.pt`
- `D:\BINN\prior_AddOn\gene_embeddings\processed\mapping_cache.tsv`
- `D:\BINN\prior_AddOn\gene_embeddings\manifest.json`

## Basic Usage

```python
from prior_AddOn.gene_embedding_loader import load_static_gene_prior

priors = load_static_gene_prior(
    source_panel=dataset.source_panel,
    species="mouse",  # or "human"
    root=r"D:\BINN\prior_AddOn\gene_embeddings",
)
```

The order of `source_panel` is preserved exactly. `priors[model]["embeddings"][i]` always corresponds to `source_panel[i]`.

## Arguments

```python
load_static_gene_prior(
    source_panel,
    species,
    models=("scgpt", "geneformer"),
    root=None,
    dataset_key=None,
    write_aligned=False,
    allow_network=True,
    use_aligned_cache=True,
    write_aligned_cache=True,
    precache_orthologs=True,
    retry_failed_mappings=False,
)
```

- `source_panel`: Gene or feature list, usually `dataset.source_panel`.
- `species`: `"human"` or `"mouse"`; aliases like `"homo_sapiens"` and `"mus_musculus"` are accepted.
- `models`: Models to load. Defaults to both `"scgpt"` and `"geneformer"`.
- `root`: Artifact root. Defaults to `prior_AddOn/gene_embeddings`.
- `dataset_key`: Optional readable cache key. If set, aligned caches use `processed/aligned/{dataset_key}_{model}.pt`.
- `write_aligned`: Backward-compatible flag. Setting it to `True` writes aligned cache.
- `allow_network`: Whether Ensembl REST API calls are allowed for mouse-to-human ortholog mapping.
- `use_aligned_cache`: If `True`, load an existing aligned cache before reading static model files.
- `write_aligned_cache`: If `True`, save aligned cache after a full build.
- `precache_orthologs`: If `True`, mouse panels are ortholog-mapped before embedding alignment.
- `retry_failed_mappings`: If `False`, cached `ensembl_request_failed:*` rows are reused. If `True`, those failed rows are queried again.

## Execution Flow

1. Try aligned cache first.
   If all requested models have valid aligned caches, the function returns immediately. It will not load the large static embedding files and will not call Ensembl.

2. If aligned cache is missing, load the needed static embedding files.
   For mouse panels, Geneformer dictionaries are also loaded because mouse genes are mapped to human Ensembl ids before model-specific matching.

3. For `species="mouse"`, pre-cache ortholog mapping for the whole input panel.
   Each unique mouse gene is resolved once. Every result is written immediately to `processed/mapping_cache.tsv`, so interrupted runs can resume from the last completed gene.

4. Align each requested model.
   scGPT and Geneformer use the pre-cached mapping results and do not trigger duplicate Ensembl calls inside each model loop.

5. Save aligned cache.
   The aligned cache stores `source_panel`, `embeddings`, `found_mask`, `mapping_table`, and `coverage`.

## Cache Files

Ortholog mapping cache:

- `processed/mapping_cache.tsv`

Aligned embedding cache with `dataset_key`:

- `processed/aligned/{dataset_key}_scgpt.pt`
- `processed/aligned/{dataset_key}_geneformer.pt`

Aligned embedding cache without `dataset_key`:

- `processed/aligned/auto_{species}_{hash}_scgpt.pt`
- `processed/aligned/auto_{species}_{hash}_geneformer.pt`

The auto hash is based on `species + source_panel`. If the gene list content or order changes, the old cache is not reused.

## Return Structure

```python
{
    "scgpt": {
        "embeddings": Tensor[n_genes, 512],
        "found_mask": BoolTensor[n_genes],
        "mapping_table": list[dict],
        "coverage": dict,
    },
    "geneformer": {
        "embeddings": Tensor[n_genes, 1152],
        "found_mask": BoolTensor[n_genes],
        "mapping_table": list[dict],
        "coverage": dict,
    },
}
```

- `embeddings`: Aligned tensor. Missing genes receive zero vectors.
- `found_mask`: Boolean mask for genes with an embedding.
- `mapping_table`: Per-gene mapping records.
- `coverage`: Summary counts and coverage ratio.

## Mapping Status Values

- `mapped`: The gene was resolved and an embedding was found.
- `missing_embedding`: The gene was resolved, but the model vocab has no embedding for it.
- `unmapped`: No usable mapping was found.
- `ambiguous`: Multiple mappings were found; the loader does not choose randomly.
- `non_gene_feature`: The input looks like an ATAC peak or special token.
- `invalid_input`: The input is empty or invalid.

## Offline Mode

```python
priors = load_static_gene_prior(
    dataset.source_panel,
    species="mouse",
    root=r"D:\BINN\prior_AddOn\gene_embeddings",
    allow_network=False,
)
```

In offline mode, mouse-to-human mapping only uses `mapping_cache.tsv`. Uncached mouse genes are marked as `unmapped`.

## Inspect Missing Genes

```python
for row in priors["geneformer"]["mapping_table"]:
    if row["status"] != "mapped":
        print(row["input_gene"], row["status"], row["reason"])
```

