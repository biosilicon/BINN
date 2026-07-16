from __future__ import annotations

import gc
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import mudata
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from datasets.h5mu_dataset import (
    H5MuDataManager,
    H5MuValidationError,
    validate_h5mu,
)
from prior_AddOn.gene_prior_filter import filter_dataset_by_gene_prior
from utils.utils_h5mu_dataloader import h5mu_dataloader


def _write_h5mu(
    path: Path,
    *,
    global_obs: list[str],
    rna_obs: list[str] | None = None,
    protein_obs: list[str] | None = None,
    rna_vars: tuple[str, ...] = ("g1", "g2"),
    protein_vars: tuple[str, ...] = ("p1",),
    rna_x: np.ndarray | None = None,
    protein_x: np.ndarray | None = None,
    sample_ids: list[str] | None = None,
    spatial: np.ndarray | None = None,
    pairing_type: str = "same_unit",
    include_sample_id: bool = True,
    include_spatial: bool = True,
    include_database: bool = True,
) -> Path:
    rna_obs = list(global_obs if rna_obs is None else rna_obs)
    protein_obs = list(global_obs if protein_obs is None else protein_obs)
    if rna_x is None:
        rna_x = np.arange(1, len(rna_obs) * len(rna_vars) + 1, dtype=np.float32).reshape(
            len(rna_obs), len(rna_vars)
        )
    if protein_x is None:
        protein_x = np.arange(
            1, len(protein_obs) * len(protein_vars) + 1, dtype=np.float32
        ).reshape(len(protein_obs), len(protein_vars))

    rna = ad.AnnData(
        X=sparse.csr_matrix(rna_x),
        obs=pd.DataFrame(index=pd.Index(rna_obs, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(rna_vars, name="feature_id")),
    )
    protein = ad.AnnData(
        X=sparse.csr_matrix(protein_x),
        obs=pd.DataFrame(index=pd.Index(protein_obs, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(protein_vars, name="feature_id")),
    )
    rna.uns["assay"] = {"technology": "test", "value_type": "counts"}
    protein.uns["assay"] = {"technology": "test", "value_type": "intensity"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mudata.MuData({"rna": rna, "protein": protein})

    self_order = list(map(str, mdata.obs_names))
    if self_order != global_obs:
        raise AssertionError(f"unexpected MuData obs order: {self_order} != {global_obs}")

    if include_sample_id:
        sample_ids = sample_ids or ["sample_01"] * len(global_obs)
        mdata.obs["sample_id"] = np.asarray(sample_ids, dtype=object)
    if include_spatial:
        if spatial is None:
            spatial = np.stack(
                [np.arange(len(global_obs), dtype=np.float32), np.zeros(len(global_obs))],
                axis=1,
            )
        mdata.obsm["spatial"] = np.asarray(spatial, dtype=np.float32)
    if include_database:
        mdata.uns["database"] = {
            "schema_version": "1.0",
            "dataset_id": path.stem,
            "source": "unit-test",
            "organism": "Homo sapiens",
            "tissue": "test tissue",
            "spatial_unit": "cell",
            "coordinate_unit": "micrometer",
            "pairing_type": pairing_type,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata.write_h5mu(path)
    return path


class H5MuDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir.name)
        self._managers: list[H5MuDataManager] = []

    def tearDown(self) -> None:
        for manager in self._managers:
            manager.close()
        gc.collect()
        self._temp_dir.cleanup()

    def _manager(self, train_path: Path, test_path: Path, **kwargs) -> H5MuDataManager:
        manager = H5MuDataManager(train_path, test_path, **kwargs)
        self._managers.append(manager)
        return manager

    def test_independent_files_preserve_boundaries_and_sample_neighbors(self) -> None:
        train_obs = [f"train_{i}" for i in range(6)]
        test_obs = [f"test_{i}" for i in range(4)]
        train_path = _write_h5mu(
            self.temp_dir / "train.h5mu",
            global_obs=train_obs,
            sample_ids=["s1"] * 3 + ["s2"] * 3,
            spatial=np.asarray(
                [[0, 0], [1, 0], [2, 0], [0, 0], [1, 0], [2, 0]],
                dtype=np.float32,
            ),
            rna_x=np.asarray(
                [[1, 0], [2, 0], [3, 0], [100, 0], [101, 0], [102, 0]],
                dtype=np.float32,
            ),
        )
        test_path = _write_h5mu(
            self.temp_dir / "test.h5mu",
            global_obs=test_obs,
            sample_ids=["s3"] * 4,
        )

        manager = self._manager(
            train_path, test_path, n_neighbors=2, preprocess="raw"
        )
        self.assertEqual(len(manager.training), 6)
        self.assertEqual(len(manager.testing), 4)
        self.assertTrue(all(name.startswith("train_") for name in manager.training.obs_names))
        self.assertTrue(all(name.startswith("test_") for name in manager.testing.obs_names))

        _, _, first_neighbors, _ = manager.training[0]
        self.assertEqual(set(first_neighbors[:, 0].tolist()), {2.0, 3.0})
        _, _, fourth_neighbors, _ = manager.training[3]
        self.assertEqual(set(fourth_neighbors[:, 0].tolist()), {101.0, 102.0})

    def test_auto_preprocessing_matches_xenium_behavior(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "auto.h5mu",
            global_obs=["a", "b"],
            rna_x=np.asarray([[1, 3], [0, 0]], dtype=np.float32),
            protein_x=np.asarray([[3], [0]], dtype=np.float32),
        )
        manager = self._manager(path, path, n_neighbors=1, preprocess="auto")
        source, target, _, _ = manager.training[0]
        expected_source = torch.log1p(torch.tensor([250.0, 750.0]))
        torch.testing.assert_close(source, expected_source)
        torch.testing.assert_close(target, torch.tensor([np.log1p(3)], dtype=torch.float32))

    def test_partial_pairing_uses_source_only_observations_as_neighbors(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "partial.h5mu",
            global_obs=["a", "b", "c"],
            rna_obs=["a", "b", "c"],
            protein_obs=["a", "c"],
            rna_vars=("g1",),
            rna_x=np.asarray([[1], [2], [3]], dtype=np.float32),
            protein_x=np.asarray([[10], [30]], dtype=np.float32),
            pairing_type="partially_shared",
        )
        manager = self._manager(path, path, n_neighbors=2, preprocess="raw")
        self.assertEqual(list(manager.training.obs_names), ["a", "c"])
        _, _, neighbors, _ = manager.training[0]
        self.assertEqual(set(neighbors[:, 0].tolist()), {2.0, 3.0})

    def test_unpaired_input_is_rejected(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "unpaired.h5mu",
            global_obs=["a", "b", "c", "d"],
            rna_obs=["a", "b"],
            protein_obs=["c", "d"],
            pairing_type="unpaired",
        )
        with self.assertRaisesRegex(ValueError, "pairing_type='unpaired'"):
            H5MuDataManager(path, path)

    def test_neighbor_padding_uses_zero_vectors(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "padding.h5mu",
            global_obs=["only"],
            sample_ids=["single"],
        )
        manager = self._manager(path, path, n_neighbors=3, preprocess="raw")
        _, _, neighbors, _ = manager.training[0]
        self.assertEqual(tuple(neighbors.shape), (3, 2))
        self.assertTrue(torch.equal(neighbors, torch.zeros_like(neighbors)))

    def test_feature_panels_must_match_names_and_order(self) -> None:
        train_path = _write_h5mu(
            self.temp_dir / "panel_train.h5mu", global_obs=["a", "b"]
        )
        order_path = _write_h5mu(
            self.temp_dir / "panel_order.h5mu",
            global_obs=["c", "d"],
            rna_vars=("g2", "g1"),
        )
        with self.assertRaisesRegex(ValueError, "order mismatches"):
            H5MuDataManager(train_path, order_path)

        names_path = _write_h5mu(
            self.temp_dir / "panel_names.h5mu",
            global_obs=["e", "f"],
            rna_vars=("g1", "g3"),
        )
        with self.assertRaisesRegex(ValueError, "missing from testing"):
            H5MuDataManager(train_path, names_path)

    def test_validation_aggregates_schema_errors(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "invalid.h5mu",
            global_obs=["a", "b"],
            include_sample_id=False,
            include_spatial=False,
            include_database=False,
        )
        with self.assertRaises(H5MuValidationError) as context:
            validate_h5mu(path)
        message = str(context.exception)
        self.assertIn("sample_id", message)
        self.assertIn("spatial", message)
        self.assertIn("database", message)

    def test_lazy_source_feature_filter_and_prior_hook(self) -> None:
        path = _write_h5mu(
            self.temp_dir / "filter.h5mu", global_obs=["a", "b", "c"]
        )
        manager = self._manager(path, path, n_neighbors=1, preprocess="raw")
        priors = {
            "test-prior": {
                "found_mask": np.asarray([True, False]),
                "embeddings": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                "mapping_table": [
                    {"feature": "g1", "status": "found"},
                    {"feature": "g2", "status": "missing"},
                ],
                "coverage": {"species": "human"},
            }
        }
        manager, filtered, info = filter_dataset_by_gene_prior(
            manager, priors, prior_model="test-prior"
        )
        self.assertEqual(manager.source_panel.tolist(), ["g1"])
        self.assertEqual(manager.source_length, 1)
        self.assertEqual(manager.rna_length, 1)
        self.assertEqual(filtered["test-prior"]["embeddings"].shape, (1, 2))
        self.assertEqual(info["removed_genes"], ["g2"])
        source, _, neighbors, _ = manager.training[0]
        self.assertEqual(tuple(source.shape), (1,))
        self.assertEqual(tuple(neighbors.shape), (1, 1))

    def test_dataloader_shapes_single_and_multiple_workers(self) -> None:
        train_path = _write_h5mu(
            self.temp_dir / "loader_train.h5mu",
            global_obs=[f"train_{i}" for i in range(8)],
        )
        test_path = _write_h5mu(
            self.temp_dir / "loader_test.h5mu",
            global_obs=[f"test_{i}" for i in range(5)],
        )
        manager = self._manager(
            train_path, test_path, n_neighbors=3, preprocess="raw"
        )

        for workers in (0, 2):
            args = SimpleNamespace(train_batch=4, test_batch=3, workers=workers)
            trainloader, testloader = h5mu_dataloader(args, manager)
            train_batch = next(iter(trainloader))
            test_batch = next(iter(testloader))
            self.assertEqual(tuple(train_batch[0].shape), (4, 2))
            self.assertEqual(tuple(train_batch[1].shape), (4, 1))
            self.assertEqual(tuple(train_batch[2].shape), (4, 3, 2))
            self.assertEqual(len(train_batch[3]), 4)
            self.assertEqual(tuple(test_batch[0].shape), (3, 2))
            del trainloader, testloader
            gc.collect()


class H5MuRealFileSmokeTest(unittest.TestCase):
    def test_rcc_example_batch(self) -> None:
        path = Path(__file__).resolve().parents[2] / "xenium_human_rcc_ffpe_rna_protein.h5mu"
        if not path.is_file():
            self.skipTest("RCC example H5MU is not available")

        manager = H5MuDataManager(path, path, n_neighbors=12, preprocess="auto")
        self.addCleanup(manager.close)
        self.assertEqual(len(manager.training), 465_534)
        self.assertEqual(manager.source_length, 405)
        self.assertEqual(manager.target_length, 27)
        source, target, neighbors, obs_id = manager.training[0]
        self.assertEqual(tuple(source.shape), (405,))
        self.assertEqual(tuple(target.shape), (27,))
        self.assertEqual(tuple(neighbors.shape), (12, 405))
        self.assertIsInstance(obs_id, str)
        self.assertTrue(torch.isfinite(source).all())
        self.assertTrue(torch.isfinite(target).all())


if __name__ == "__main__":
    unittest.main()
