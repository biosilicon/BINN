from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by the training path.
    torch = None


DATASET_SOURCE_FIELDS = {
    "AD_Mouse": (0, 3),
    "Breast_cancer": (0, 3),
    "SMA": (1, 3),
}


def filter_dataset_by_gene_prior(
    dataset: Any,
    priors: dict[str, dict[str, Any]],
    prior_model: str | None = None,
    source_index: int | None = None,
    neighbor_index: int | None = None,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]:
    """Filter a prepared dataset to genes found in the selected static prior.

    The function mutates and returns ``dataset`` so downstream notebook cells can
    use one explicit assignment and avoid stale dataloaders with old dimensions.
    """

    resolved_model = _resolve_prior_model(priors, prior_model)
    selected_prior = priors[resolved_model]
    if "found_mask" not in selected_prior:
        raise ValueError(f"priors[{resolved_model!r}] does not contain 'found_mask'.")

    original_source_panel = list(dataset.source_panel)
    keep_mask = _as_bool_numpy(selected_prior["found_mask"])
    if keep_mask.ndim != 1:
        raise ValueError("'found_mask' must be one-dimensional.")
    if len(keep_mask) != len(original_source_panel):
        raise ValueError(
            "Length mismatch between dataset.source_panel "
            f"({len(original_source_panel)}) and priors[{resolved_model!r}]['found_mask'] "
            f"({len(keep_mask)})."
        )
    if not keep_mask.any():
        raise ValueError(
            f"No genes in dataset.source_panel were found in prior model {resolved_model!r}."
        )

    source_index, neighbor_index = _resolve_feature_indices(dataset, source_index, neighbor_index)
    filtered_source_panel = [
        gene for gene, keep in zip(original_source_panel, keep_mask.tolist()) if keep
    ]
    removed_genes = [
        gene for gene, keep in zip(original_source_panel, keep_mask.tolist()) if not keep
    ]

    dataset.source_panel = _filter_source_panel(dataset.source_panel, keep_mask, filtered_source_panel)
    _update_length_attributes(dataset, len(filtered_source_panel))
    _update_rna_mask(dataset, keep_mask, len(original_source_panel))
    _filter_dataset_splits(dataset, keep_mask, source_index, neighbor_index)

    filtered_priors = _filter_priors(priors, keep_mask)
    removed_mapping_table = [
        row
        for row, keep in zip(selected_prior.get("mapping_table", []), keep_mask.tolist())
        if not keep
    ]

    filter_info = {
        "prior_model": resolved_model,
        "original_source_panel": original_source_panel,
        "filtered_source_panel": filtered_source_panel,
        "removed_genes": removed_genes,
        "removed_mapping_table": removed_mapping_table,
        "keep_mask": keep_mask,
        "coverage_before": deepcopy(selected_prior.get("coverage")),
        "coverage_after": deepcopy(filtered_priors[resolved_model].get("coverage")),
    }
    return dataset, filtered_priors, filter_info


def _resolve_prior_model(
    priors: dict[str, dict[str, Any]],
    prior_model: str | None,
) -> str:
    if not priors:
        raise ValueError("'priors' must contain at least one prior model.")
    if prior_model is not None:
        if prior_model not in priors:
            raise ValueError(
                f"prior_model {prior_model!r} was not found in priors. "
                f"Available models: {sorted(priors)}"
            )
        return prior_model
    if len(priors) == 1:
        return next(iter(priors))
    raise ValueError(
        "Multiple prior models were provided. Pass prior_model explicitly. "
        f"Available models: {sorted(priors)}"
    )


def _resolve_feature_indices(
    dataset: Any,
    source_index: int | None,
    neighbor_index: int | None,
) -> tuple[int, int]:
    if source_index is not None and neighbor_index is not None:
        return source_index, neighbor_index
    class_name = dataset.__class__.__name__
    if class_name in DATASET_SOURCE_FIELDS:
        return DATASET_SOURCE_FIELDS[class_name]
    raise ValueError(
        f"Unknown dataset class {class_name!r}. Pass source_index and neighbor_index "
        "so source features and source-neighbor features can be filtered safely."
    )


def _as_bool_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=bool)


def _filter_source_panel(source_panel: Any, keep_mask: np.ndarray, fallback: list[Any]) -> Any:
    try:
        return source_panel[keep_mask]
    except Exception:
        try:
            return source_panel.__class__(fallback)
        except Exception:
            return fallback


def _update_length_attributes(dataset: Any, filtered_length: int) -> None:
    for attr in ("rna_length", "source_length", "source_len", "rna_len"):
        if hasattr(dataset, attr):
            setattr(dataset, attr, filtered_length)


def _update_rna_mask(dataset: Any, keep_mask: np.ndarray, original_length: int) -> None:
    if not hasattr(dataset, "rna_mask"):
        return
    rna_mask = np.asarray(dataset.rna_mask)
    if rna_mask.dtype != bool or int(rna_mask.sum()) != original_length:
        return
    updated = rna_mask.copy()
    updated[np.flatnonzero(rna_mask)] = keep_mask
    dataset.rna_mask = updated


def _filter_dataset_splits(
    dataset: Any,
    keep_mask: np.ndarray,
    source_index: int,
    neighbor_index: int,
) -> None:
    for split_name in ("training", "testing", "val"):
        if hasattr(dataset, split_name):
            split = getattr(dataset, split_name)
            setattr(
                dataset,
                split_name,
                [
                    _filter_sample(sample, keep_mask, source_index, neighbor_index)
                    for sample in split
                ],
            )


def _filter_sample(
    sample: Any,
    keep_mask: np.ndarray,
    source_index: int,
    neighbor_index: int,
) -> Any:
    values = list(sample)
    values[source_index] = _filter_feature_array(values[source_index], keep_mask)
    values[neighbor_index] = _filter_feature_array(values[neighbor_index], keep_mask)
    return tuple(values)


def _filter_feature_array(value: Any, keep_mask: np.ndarray) -> Any:
    if torch is not None and torch.is_tensor(value):
        mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=value.device)
        return value[..., mask]
    return np.asarray(value)[..., keep_mask]


def _filter_priors(
    priors: dict[str, dict[str, Any]],
    keep_mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    keep_list = keep_mask.tolist()
    for model_name, model_prior in priors.items():
        filtered_prior = dict(model_prior)
        if "embeddings" in model_prior:
            filtered_prior["embeddings"] = _filter_first_dim(model_prior["embeddings"], keep_mask)
        if "found_mask" in model_prior:
            filtered_prior["found_mask"] = _filter_first_dim(model_prior["found_mask"], keep_mask)
        if "mapping_table" in model_prior:
            filtered_prior["mapping_table"] = [
                row for row, keep in zip(model_prior["mapping_table"], keep_list) if keep
            ]
        if "found_mask" in filtered_prior and "mapping_table" in filtered_prior:
            species = None
            coverage = model_prior.get("coverage")
            if isinstance(coverage, dict):
                species = coverage.get("species")
            filtered_prior["coverage"] = _coverage(
                filtered_prior["found_mask"],
                filtered_prior["mapping_table"],
                model_name,
                species,
            )
        filtered[model_name] = filtered_prior
    return filtered


def _filter_first_dim(value: Any, keep_mask: np.ndarray) -> Any:
    if torch is not None and torch.is_tensor(value):
        mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=value.device)
        return value[mask]
    return np.asarray(value)[keep_mask]


def _coverage(
    found_mask: Any,
    mapping_rows: list[dict[str, Any]],
    model_name: str,
    species: str | None,
) -> dict[str, Any]:
    found_mask_np = _as_bool_numpy(found_mask)
    total = len(mapping_rows)
    found = int(found_mask_np.sum()) if total else 0
    by_status: dict[str, int] = {}
    for row in mapping_rows:
        status = str(row.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "model": model_name,
        "species": species,
        "n_features": total,
        "n_found": found,
        "coverage": (found / total) if total else 0.0,
        "by_status": by_status,
    }
