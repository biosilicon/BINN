from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import mudata
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


PreprocessMode = Literal["auto", "raw"]
TensorTransform = Callable[[torch.Tensor], torch.Tensor]

_REQUIRED_DATABASE_FIELDS = {
    "schema_version",
    "dataset_id",
    "source",
    "organism",
    "tissue",
    "spatial_unit",
    "coordinate_unit",
    "pairing_type",
}
_SUPPORTED_PAIRING_TYPES = {"same_unit", "partially_shared", "unpaired"}
_LOG1P_VALUE_TYPES = {"intensity", "background_corrected_intensity"}


class H5MuValidationError(ValueError):
    """Raised when an H5MU file does not satisfy the database schema."""


@dataclass(frozen=True)
class H5MuFileMetadata:
    """Small, in-memory description of a validated H5MU file."""

    path: str
    n_obs: int
    modalities: tuple[str, ...]
    modality_shapes: dict[str, tuple[int, int]]
    samples: tuple[str, ...]
    spatial_dimensions: int
    database: dict[str, Any]
    assays: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class _H5MuIndex:
    metadata: H5MuFileMetadata
    obs_names: np.ndarray
    sample_ids: np.ndarray
    spatial: np.ndarray
    modality_obs_names: dict[str, np.ndarray]
    modality_var_names: dict[str, np.ndarray]


def validate_h5mu(path: str | os.PathLike[str]) -> H5MuFileMetadata:
    """Validate ``path`` against the minimal spatial multi-omics schema.

    All discoverable schema problems are collected and reported in one
    :class:`H5MuValidationError`. The returned metadata does not keep the HDF5
    file open.
    """

    return _inspect_h5mu(path).metadata


class H5MuDataset(Dataset):
    """Map-style, backed PyTorch dataset for paired spatial modalities.

    Instances are normally created by :class:`H5MuDataManager`. Each sample is
    returned as ``(source, target, source_neighbors, obs_id)``. HDF5 handles are
    opened lazily and are never serialized into DataLoader workers.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str],
        source_modality: str,
        target_modality: str,
        source_rows: np.ndarray,
        target_rows: np.ndarray,
        neighbor_source_rows: np.ndarray,
        obs_names: np.ndarray,
        sample_ids: np.ndarray,
        spatial: np.ndarray,
        source_n_vars: int,
        target_n_vars: int,
        source_value_type: str,
        target_value_type: str,
        preprocess: PreprocessMode = "auto",
        rna_target_sum: float = 1e3,
        source_transform: TensorTransform | None = None,
        target_transform: TensorTransform | None = None,
    ) -> None:
        if preprocess not in {"auto", "raw"}:
            raise ValueError("preprocess must be either 'auto' or 'raw'.")
        if rna_target_sum <= 0:
            raise ValueError("rna_target_sum must be positive.")

        self.path = str(Path(path).resolve())
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.preprocess = preprocess
        self.rna_target_sum = float(rna_target_sum)
        self.source_transform = source_transform
        self.target_transform = target_transform

        self._source_rows = np.asarray(source_rows, dtype=np.int64)
        self._target_rows = np.asarray(target_rows, dtype=np.int64)
        self._neighbor_source_rows = np.asarray(neighbor_source_rows)
        self.obs_names = np.asarray(obs_names, dtype=object)
        self.sample_ids = np.asarray(sample_ids, dtype=object)
        self.spatial = np.asarray(spatial, dtype=np.float32)
        self._source_n_vars = int(source_n_vars)
        self._target_n_vars = int(target_n_vars)
        self._source_feature_indices = np.arange(source_n_vars, dtype=np.int64)
        self.source_value_type = str(source_value_type).lower()
        self.target_value_type = str(target_value_type).lower()

        n_obs = len(self._source_rows)
        expected_shapes = {
            "target_rows": (n_obs,),
            "neighbor_source_rows": (n_obs, self._neighbor_source_rows.shape[1]),
            "obs_names": (n_obs,),
            "sample_ids": (n_obs,),
        }
        actual_shapes = {
            "target_rows": self._target_rows.shape,
            "neighbor_source_rows": self._neighbor_source_rows.shape,
            "obs_names": self.obs_names.shape,
            "sample_ids": self.sample_ids.shape,
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(
                    f"{name} has shape {actual_shapes[name]}, expected {expected}."
                )
        if self.spatial.shape[0] != n_obs:
            raise ValueError("spatial and source_rows must contain the same observations.")

        self._mdata = None
        self._open_pid: int | None = None

    @property
    def source_length(self) -> int:
        return len(self._source_feature_indices)

    @property
    def target_length(self) -> int:
        return self._target_n_vars

    @property
    def n_neighbors(self) -> int:
        return self._neighbor_source_rows.shape[1]

    def __len__(self) -> int:
        return len(self._source_rows)

    def __getitem__(self, index: int):
        index = _normalize_index(index, len(self))
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]):
        """Load all rows needed by a DataLoader batch in two HDF5 reads."""

        normalized = np.asarray(
            [_normalize_index(index, len(self)) for index in indices], dtype=np.int64
        )
        if normalized.size == 0:
            return []

        mdata = self._ensure_open()
        center_source_rows = self._source_rows[normalized]
        neighbor_rows = np.asarray(
            self._neighbor_source_rows[normalized], dtype=np.int64
        )
        valid_neighbors = neighbor_rows >= 0

        requested_source_rows = np.concatenate(
            [center_source_rows, neighbor_rows[valid_neighbors]]
        )
        source_values = _read_backed_rows(
            mdata.mod[self.source_modality].X, requested_source_rows
        )
        source_values = _preprocess_values(
            source_values,
            value_type=self.source_value_type,
            mode=self.preprocess,
            counts_target_sum=self.rna_target_sum,
            modality=self.source_modality,
        )
        source_values = source_values[:, self._source_feature_indices]

        batch_size = len(normalized)
        centers = source_values[:batch_size]
        neighbors = np.zeros(
            (batch_size, self.n_neighbors, self.source_length), dtype=np.float32
        )
        neighbors.reshape(-1, self.source_length)[valid_neighbors.reshape(-1)] = (
            source_values[batch_size:]
        )

        target_values = _read_backed_rows(
            mdata.mod[self.target_modality].X, self._target_rows[normalized]
        )
        target_values = _preprocess_values(
            target_values,
            value_type=self.target_value_type,
            mode=self.preprocess,
            counts_target_sum=self.rna_target_sum,
            modality=self.target_modality,
        )

        center_tensors = torch.from_numpy(np.ascontiguousarray(centers))
        neighbor_tensors = torch.from_numpy(np.ascontiguousarray(neighbors))
        target_tensors = torch.from_numpy(np.ascontiguousarray(target_values))

        center_tensors = _apply_transform(
            self.source_transform, center_tensors, "source_transform"
        )
        neighbor_tensors = _apply_transform(
            self.source_transform, neighbor_tensors, "source_transform"
        )
        target_tensors = _apply_transform(
            self.target_transform, target_tensors, "target_transform"
        )

        return [
            (
                center_tensors[i],
                target_tensors[i],
                neighbor_tensors[i],
                str(self.obs_names[index]),
            )
            for i, index in enumerate(normalized)
        ]

    def select_source_features(self, keep_mask: Sequence[bool]) -> None:
        """Apply a lazy source-feature mask without materializing observations."""

        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or len(keep_mask) != self.source_length:
            raise ValueError(
                "keep_mask must be one-dimensional and match the current source panel "
                f"length ({self.source_length})."
            )
        if not keep_mask.any():
            raise ValueError("keep_mask must retain at least one source feature.")
        self._source_feature_indices = self._source_feature_indices[keep_mask]

    def close(self) -> None:
        mdata = getattr(self, "_mdata", None)
        if mdata is not None:
            try:
                mdata.file.close()
            except (AttributeError, OSError):
                pass
        self._mdata = None
        self._open_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mdata"] = None
        state["_open_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _ensure_open(self):
        pid = os.getpid()
        if self._mdata is None or self._open_pid != pid:
            self.close()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._mdata = mudata.read_h5mu(self.path, backed="r")
            self._open_pid = pid
        return self._mdata


class H5MuDataManager:
    """Manage independent training and testing H5MU files."""

    def __init__(
        self,
        train_path: str | os.PathLike[str],
        test_path: str | os.PathLike[str],
        source_modality: str = "rna",
        target_modality: str = "protein",
        n_neighbors: int = 12,
        preprocess: PreprocessMode = "auto",
        rna_target_sum: float = 1e3,
        source_transform: TensorTransform | None = None,
        target_transform: TensorTransform | None = None,
    ) -> None:
        if source_modality == target_modality:
            raise ValueError("source_modality and target_modality must be different.")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")

        train_index = _inspect_h5mu(train_path)
        same_input_file = Path(test_path).expanduser().resolve() == Path(
            train_path
        ).expanduser().resolve()
        if same_input_file:
            test_index = train_index
        else:
            test_index = _inspect_h5mu(test_path)
        _require_modality(train_index, source_modality, "training")
        _require_modality(train_index, target_modality, "training")
        _require_modality(test_index, source_modality, "testing")
        _require_modality(test_index, target_modality, "testing")

        train_source_panel = train_index.modality_var_names[source_modality]
        train_target_panel = train_index.modality_var_names[target_modality]
        _require_identical_panel(
            train_source_panel,
            test_index.modality_var_names[source_modality],
            f"source modality {source_modality!r}",
        )
        _require_identical_panel(
            train_target_panel,
            test_index.modality_var_names[target_modality],
            f"target modality {target_modality!r}",
        )

        self.train_path = train_index.metadata.path
        self.test_path = test_index.metadata.path
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.n_neighbors = int(n_neighbors)
        self.preprocess = preprocess
        self.rna_target_sum = float(rna_target_sum)

        self.train_metadata = train_index.metadata
        self.test_metadata = test_index.metadata
        self.train_database = dict(train_index.metadata.database)
        self.test_database = dict(test_index.metadata.database)
        self.train_assays = dict(train_index.metadata.assays)
        self.test_assays = dict(test_index.metadata.assays)
        self.train_sample_ids = train_index.metadata.samples
        self.test_sample_ids = test_index.metadata.samples

        self.source_panel = train_source_panel.copy()
        self.target_panel = train_target_panel.copy()
        self.source_length = len(self.source_panel)
        self.target_length = len(self.target_panel)

        self.training = _build_split_dataset(
            index=train_index,
            source_modality=source_modality,
            target_modality=target_modality,
            n_neighbors=n_neighbors,
            preprocess=preprocess,
            rna_target_sum=rna_target_sum,
            source_transform=source_transform,
            target_transform=target_transform,
        )
        self.testing = _build_split_dataset(
            index=test_index,
            source_modality=source_modality,
            target_modality=target_modality,
            n_neighbors=n_neighbors,
            preprocess=preprocess,
            rna_target_sum=rna_target_sum,
            source_transform=source_transform,
            target_transform=target_transform,
        )
        self.train_dataset = self.training
        self.test_dataset = self.testing
        self._sync_modality_length_aliases()

    def select_source_features(self, keep_mask: Sequence[bool]) -> None:
        """Filter source features in both splits while preserving backed I/O."""

        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or len(keep_mask) != self.source_length:
            raise ValueError(
                "keep_mask must be one-dimensional and match dataset.source_panel "
                f"length ({self.source_length})."
            )
        if not keep_mask.any():
            raise ValueError("keep_mask must retain at least one source feature.")

        self.training.select_source_features(keep_mask)
        self.testing.select_source_features(keep_mask)
        self.source_panel = self.source_panel[keep_mask]
        self.source_length = len(self.source_panel)
        self._sync_modality_length_aliases()

    def close(self) -> None:
        self.training.close()
        self.testing.close()

    def _sync_modality_length_aliases(self) -> None:
        if self.source_modality == "rna":
            self.rna_length = self.source_length
        elif self.target_modality == "rna":
            self.rna_length = self.target_length

        if self.source_modality == "protein":
            self.protein_length = self.source_length
        elif self.target_modality == "protein":
            self.protein_length = self.target_length


def _inspect_h5mu(path: str | os.PathLike[str]) -> _H5MuIndex:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H5MU file not found: {resolved}")

    errors: list[str] = []
    mdata = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            mdata = mudata.read_h5mu(resolved, backed="r")

        modalities = tuple(mdata.mod.keys())
        if len(modalities) < 2:
            errors.append("the file must contain at least two modalities")

        if mdata.n_obs <= 0:
            errors.append("top-level obs must contain at least one observation")
        if not mdata.obs_names.is_unique:
            errors.append("top-level obs_names must be unique")
        if _contains_empty_names(mdata.obs_names):
            errors.append("top-level obs_names must not contain empty IDs")

        if "sample_id" not in mdata.obs.columns:
            errors.append("top-level obs is missing required column 'sample_id'")
            sample_ids = np.full(mdata.n_obs, "", dtype=object)
        else:
            sample_series = mdata.obs["sample_id"]
            if sample_series.isna().any():
                errors.append("top-level obs['sample_id'] contains missing values")
            sample_ids = sample_series.astype(str).to_numpy(dtype=object)
            if any(not value.strip() for value in sample_ids):
                errors.append("top-level obs['sample_id'] contains empty values")

        if "spatial" not in mdata.obsm:
            errors.append("top-level obsm is missing required matrix 'spatial'")
            spatial = np.empty((mdata.n_obs, 0), dtype=np.float32)
        else:
            spatial = np.asarray(mdata.obsm["spatial"])
            if spatial.ndim != 2 or spatial.shape[0] != mdata.n_obs:
                errors.append(
                    "obsm['spatial'] must be a two-dimensional matrix with one row "
                    "per top-level observation"
                )
            if spatial.ndim != 2 or spatial.shape[1] not in (2, 3):
                errors.append("obsm['spatial'] must have two or three columns")
            if not np.issubdtype(spatial.dtype, np.number):
                errors.append("obsm['spatial'] must use a numeric dtype")
            elif not np.isfinite(spatial).all():
                errors.append("obsm['spatial'] contains non-finite coordinates")
            spatial = np.asarray(spatial, dtype=np.float32)

        database_value = mdata.uns.get("database")
        if not isinstance(database_value, dict):
            errors.append("top-level uns is missing required dictionary 'database'")
            database: dict[str, Any] = {}
        else:
            database = dict(database_value)
            missing_database = sorted(_REQUIRED_DATABASE_FIELDS - database.keys())
            if missing_database:
                errors.append(
                    "uns['database'] is missing fields: " + ", ".join(missing_database)
                )
            pairing_type = database.get("pairing_type")
            if pairing_type is not None and pairing_type not in _SUPPORTED_PAIRING_TYPES:
                errors.append(
                    "uns['database']['pairing_type'] must be one of "
                    + ", ".join(sorted(_SUPPORTED_PAIRING_TYPES))
                )

        global_obs_names = pd.Index(mdata.obs_names)
        modality_obs_names: dict[str, np.ndarray] = {}
        modality_var_names: dict[str, np.ndarray] = {}
        modality_shapes: dict[str, tuple[int, int]] = {}
        assays: dict[str, dict[str, Any]] = {}

        for modality, adata in mdata.mod.items():
            prefix = f"modality {modality!r}"
            modality_obs_names[modality] = adata.obs_names.to_numpy(dtype=object)
            modality_var_names[modality] = adata.var_names.to_numpy(dtype=object)
            modality_shapes[modality] = (adata.n_obs, adata.n_vars)

            if adata.X is None:
                errors.append(f"{prefix} is missing X")
            else:
                if adata.X.shape != (adata.n_obs, adata.n_vars):
                    errors.append(f"{prefix} X shape does not match obs/var dimensions")
                if not np.issubdtype(adata.X.dtype, np.number):
                    errors.append(f"{prefix} X must use a numeric dtype")
            if adata.n_obs <= 0:
                errors.append(f"{prefix} must contain at least one observation")
            if adata.n_vars <= 0:
                errors.append(f"{prefix} must contain at least one feature")
            if not adata.obs_names.is_unique:
                errors.append(f"{prefix} obs_names must be unique")
            if not adata.var_names.is_unique:
                errors.append(f"{prefix} var_names must be unique")
            if _contains_empty_names(adata.obs_names):
                errors.append(f"{prefix} obs_names must not contain empty IDs")
            if _contains_empty_names(adata.var_names):
                errors.append(f"{prefix} var_names must not contain empty IDs")
            if np.any(global_obs_names.get_indexer(adata.obs_names) < 0):
                errors.append(f"{prefix} obs_names must be a subset of top-level obs_names")

            assay_value = adata.uns.get("assay")
            if not isinstance(assay_value, dict):
                errors.append(f"{prefix} uns is missing required dictionary 'assay'")
                assay: dict[str, Any] = {}
            else:
                assay = dict(assay_value)
                missing_assay = [
                    field for field in ("technology", "value_type") if field not in assay
                ]
                if missing_assay:
                    errors.append(
                        f"{prefix} uns['assay'] is missing fields: "
                        + ", ".join(missing_assay)
                    )
            assays[modality] = assay

        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise H5MuValidationError(
                f"H5MU validation failed for {resolved}:\n{details}"
            )

        metadata = H5MuFileMetadata(
            path=str(resolved),
            n_obs=mdata.n_obs,
            modalities=modalities,
            modality_shapes=modality_shapes,
            samples=tuple(pd.unique(sample_ids).tolist()),
            spatial_dimensions=spatial.shape[1],
            database=database,
            assays=assays,
        )
        return _H5MuIndex(
            metadata=metadata,
            obs_names=mdata.obs_names.to_numpy(dtype=object),
            sample_ids=sample_ids,
            spatial=spatial,
            modality_obs_names=modality_obs_names,
            modality_var_names=modality_var_names,
        )
    finally:
        if mdata is not None:
            try:
                mdata.file.close()
            except (AttributeError, OSError):
                pass


def _build_split_dataset(
    *,
    index: _H5MuIndex,
    source_modality: str,
    target_modality: str,
    n_neighbors: int,
    preprocess: PreprocessMode,
    rna_target_sum: float,
    source_transform: TensorTransform | None,
    target_transform: TensorTransform | None,
) -> H5MuDataset:
    pairing_type = str(index.metadata.database["pairing_type"])
    if pairing_type == "unpaired":
        raise ValueError(
            f"{index.metadata.path} declares pairing_type='unpaired'; paired "
            "source-to-target supervision cannot be constructed."
        )

    source_names = pd.Index(index.modality_obs_names[source_modality])
    target_names = pd.Index(index.modality_obs_names[target_modality])
    target_rows_for_source = target_names.get_indexer(source_names)
    if pairing_type == "same_unit" and (
        len(source_names) != len(target_names) or np.any(target_rows_for_source < 0)
    ):
        raise ValueError(
            f"{index.metadata.path} declares pairing_type='same_unit', but "
            f"modalities {source_modality!r} and {target_modality!r} do not contain "
            "the same observation IDs."
        )

    source_rows = np.flatnonzero(target_rows_for_source >= 0).astype(np.int64)
    if source_rows.size == 0:
        raise ValueError(
            f"Modalities {source_modality!r} and {target_modality!r} in "
            f"{index.metadata.path} do not share any observation IDs."
        )
    target_rows = target_rows_for_source[source_rows].astype(np.int64)

    global_names = pd.Index(index.obs_names)
    source_global_rows = global_names.get_indexer(source_names)
    center_global_rows = source_global_rows[source_rows]
    source_sample_ids = index.sample_ids[source_global_rows]
    center_sample_ids = source_sample_ids[source_rows]
    center_spatial = index.spatial[center_global_rows]

    neighbor_rows = _build_spatial_neighbors(
        source_rows=source_rows,
        source_sample_ids=source_sample_ids,
        source_spatial=index.spatial[source_global_rows],
        n_neighbors=n_neighbors,
    )

    max_source_row = len(source_names) - 1
    neighbor_dtype = np.int32 if max_source_row <= np.iinfo(np.int32).max else np.int64
    neighbor_rows = neighbor_rows.astype(neighbor_dtype, copy=False)

    source_assay = index.metadata.assays[source_modality]
    target_assay = index.metadata.assays[target_modality]
    return H5MuDataset(
        path=index.metadata.path,
        source_modality=source_modality,
        target_modality=target_modality,
        source_rows=source_rows,
        target_rows=target_rows,
        neighbor_source_rows=neighbor_rows,
        obs_names=source_names.to_numpy(dtype=object)[source_rows],
        sample_ids=center_sample_ids,
        spatial=center_spatial,
        source_n_vars=len(index.modality_var_names[source_modality]),
        target_n_vars=len(index.modality_var_names[target_modality]),
        source_value_type=str(source_assay["value_type"]),
        target_value_type=str(target_assay["value_type"]),
        preprocess=preprocess,
        rna_target_sum=rna_target_sum,
        source_transform=source_transform,
        target_transform=target_transform,
    )


def _build_spatial_neighbors(
    *,
    source_rows: np.ndarray,
    source_sample_ids: np.ndarray,
    source_spatial: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    result = np.full((len(source_rows), n_neighbors), -1, dtype=np.int64)
    center_positions_by_sample: dict[str, list[int]] = {}
    for center_position, source_row in enumerate(source_rows):
        sample_id = str(source_sample_ids[source_row])
        center_positions_by_sample.setdefault(sample_id, []).append(center_position)

    for sample_id, center_positions_list in center_positions_by_sample.items():
        center_positions = np.asarray(center_positions_list, dtype=np.int64)
        candidate_rows = np.flatnonzero(source_sample_ids == sample_id).astype(np.int64)
        query_k = min(n_neighbors + 1, len(candidate_rows))
        if query_k <= 0:
            continue

        tree = cKDTree(source_spatial[candidate_rows])
        _, local_neighbors = tree.query(
            source_spatial[source_rows[center_positions]], k=query_k, workers=-1
        )
        if query_k == 1:
            local_neighbors = np.asarray(local_neighbors)[:, None]
        candidate_neighbor_rows = candidate_rows[np.asarray(local_neighbors)]

        is_not_self = candidate_neighbor_rows != source_rows[center_positions, None]
        neighbor_rank = np.cumsum(is_not_self, axis=1) - 1
        usable = is_not_self & (neighbor_rank < n_neighbors)
        local_center, query_column = np.nonzero(usable)
        result[
            center_positions[local_center], neighbor_rank[local_center, query_column]
        ] = candidate_neighbor_rows[local_center, query_column]

    return result


def _read_backed_rows(matrix: Any, requested_rows: np.ndarray) -> np.ndarray:
    unique_rows, inverse = np.unique(requested_rows, return_inverse=True)
    values = matrix[unique_rows, :]
    if sparse.issparse(values):
        values = values.toarray()
    else:
        values = np.asarray(values)
    return np.asarray(values[inverse], dtype=np.float32)


def _preprocess_values(
    values: np.ndarray,
    *,
    value_type: str,
    mode: PreprocessMode,
    counts_target_sum: float,
    modality: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if mode == "raw":
        return values

    if value_type == "counts":
        if np.any(values < 0):
            raise ValueError(f"{modality!r} counts contain negative values.")
        row_sums = values.sum(axis=1, keepdims=True)
        scales = np.divide(
            counts_target_sum,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float32),
            where=row_sums > 0,
        )
        return np.log1p(values * scales).astype(np.float32, copy=False)

    if value_type in _LOG1P_VALUE_TYPES:
        if np.any(values < 0):
            raise ValueError(
                f"{modality!r} {value_type} values contain negative values and "
                "cannot use automatic log1p preprocessing. Use preprocess='raw' "
                "with a custom transform instead."
            )
        return np.log1p(values).astype(np.float32, copy=False)

    return values


def _apply_transform(
    transform: TensorTransform | None, values: torch.Tensor, name: str
) -> torch.Tensor:
    if transform is None:
        return values
    original_shape = values.shape
    transformed = transform(values)
    if not torch.is_tensor(transformed):
        raise TypeError(f"{name} must return a torch.Tensor.")
    if transformed.shape != original_shape:
        raise ValueError(
            f"{name} changed tensor shape from {tuple(original_shape)} to "
            f"{tuple(transformed.shape)}; feature dimensions must remain stable."
        )
    return transformed.to(dtype=torch.float32)


def _require_modality(index: _H5MuIndex, modality: str, split_name: str) -> None:
    if modality not in index.metadata.modalities:
        raise ValueError(
            f"The {split_name} file does not contain modality {modality!r}. "
            f"Available modalities: {list(index.metadata.modalities)}"
        )


def _require_identical_panel(
    train_panel: np.ndarray, test_panel: np.ndarray, panel_label: str
) -> None:
    train_values = [str(value) for value in train_panel]
    test_values = [str(value) for value in test_panel]
    if train_values == test_values:
        return

    train_set = set(train_values)
    test_set = set(test_values)
    missing = [value for value in train_values if value not in test_set]
    extra = [value for value in test_values if value not in train_set]
    order_mismatches = [
        (i, train_value, test_value)
        for i, (train_value, test_value) in enumerate(zip(train_values, test_values))
        if train_value != test_value
    ]
    parts = [
        f"Training and testing panels for {panel_label} must have identical names "
        "in identical order."
    ]
    if missing:
        parts.append(f"missing from testing ({len(missing)}): {missing[:5]}")
    if extra:
        parts.append(f"extra in testing ({len(extra)}): {extra[:5]}")
    if not missing and not extra:
        parts.append(
            "order mismatches: "
            + str(
                [
                    {"index": i, "train": train, "test": test}
                    for i, train, test in order_mismatches[:5]
                ]
            )
        )
    raise ValueError(" ".join(parts))


def _contains_empty_names(values: pd.Index) -> bool:
    return any(not str(value).strip() for value in values)


def _normalize_index(index: int, length: int) -> int:
    index = int(index)
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError(f"dataset index {index} is out of range for length {length}")
    return index


__all__ = [
    "H5MuDataManager",
    "H5MuDataset",
    "H5MuFileMetadata",
    "H5MuValidationError",
    "validate_h5mu",
]
