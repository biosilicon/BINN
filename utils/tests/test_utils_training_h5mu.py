from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from args.args_h5mu import generate_args
from datasets.h5mu_dataset import H5MuDataManager
from datasets.tests.test_h5mu_dataset import _write_h5mu
from model.nicheTrans import NicheTrans
from utils.utils import set_seed
from utils.utils_h5mu_dataloader import h5mu_dataloader
from utils.utils_training_h5mu import (
    build_criterion,
    evaluate,
    fit,
    infer_task_type,
    train_one_epoch,
)


class _TupleDataset(Dataset):
    def __init__(self, source, target, neighbors):
        self.source = torch.as_tensor(source, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.neighbors = torch.as_tensor(neighbors, dtype=torch.float32)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index], self.neighbors[index], str(index)


class _SourceModel(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, source, source_neighbors):
        return source[:, : self.target_length]


class _CaptureNeighborModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.last_neighbors = None

    def forward(self, source, source_neighbors):
        self.last_neighbors = source_neighbors.detach().clone()
        return self.linear(source)


def _loader(source, target, neighbors, batch_size=None):
    dataset = _TupleDataset(source, target, neighbors)
    return DataLoader(dataset, batch_size=batch_size or max(len(dataset), 1))


def test_args_are_not_coupled_to_notebook_kernel_arguments():
    args = generate_args(
        ["--train-path", "train.h5mu", "--test-path", "test.h5mu", "--task", "auto"]
    )
    assert args.source_modality == "rna"
    assert args.target_modality == "protein"
    assert args.n_neighbors == 12


def test_task_inference_and_value_type_mismatch():
    assert infer_task_type("binary", "binary") == "binary"
    assert infer_task_type("intensity", "intensity") == "regression"
    assert infer_task_type("counts", "counts", requested="binary") == "binary"
    with pytest.raises(ValueError, match="must match"):
        infer_task_type("counts", "intensity")


def test_regression_metrics_skip_constant_targets():
    source = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=np.float32)
    target = np.asarray([[0, 5], [1, 5], [2, 5], [3, 5]], dtype=np.float32)
    neighbors = np.zeros((4, 2, 2), dtype=np.float32)
    result = evaluate(
        model=_SourceModel(target_length=2),
        dataloader=_loader(source, target, neighbors),
        criterion=build_criterion("regression"),
        device=torch.device("cpu"),
        task="regression",
        target_panel=["variable", "constant"],
    )
    assert result["summary"]["pearson_mean"] == pytest.approx(1.0)
    assert result["summary"]["n_valid_pearson"] == 1
    assert math.isnan(result["per_feature"][1]["pearson"])


def test_binary_metrics_skip_single_class_targets():
    logits = np.asarray([[-2, 1], [-1, 1], [1, 1], [2, 1]], dtype=np.float32)
    target = np.asarray([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
    neighbors = np.zeros((4, 2, 2), dtype=np.float32)
    result = evaluate(
        model=_SourceModel(target_length=2),
        dataloader=_loader(logits, target, neighbors),
        criterion=build_criterion("binary"),
        device=torch.device("cpu"),
        task="binary",
        target_panel=["two_class", "single_class"],
    )
    assert result["summary"]["auroc_mean"] == pytest.approx(1.0)
    assert result["summary"]["n_valid_auroc"] == 1
    assert math.isnan(result["per_feature"][1]["auroc"])


def test_neighbor_masking_can_remove_all_neighbors():
    source = np.asarray([[1], [2]], dtype=np.float32)
    target = np.asarray([[1], [2]], dtype=np.float32)
    neighbors = np.ones((2, 3, 1), dtype=np.float32)
    model = _CaptureNeighborModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_one_epoch(
        model=model,
        trainloader=_loader(source, target, neighbors, batch_size=2),
        optimizer=optimizer,
        criterion=build_criterion("regression"),
        device=torch.device("cpu"),
        task="regression",
        neighbor_mask_probability=1.0,
        neighbor_keep_probability=0.0,
    )
    assert torch.count_nonzero(model.last_neighbors) == 0


def test_empty_training_loader_is_rejected():
    dataset = _TupleDataset(
        np.empty((0, 1), dtype=np.float32),
        np.empty((0, 1), dtype=np.float32),
        np.empty((0, 2, 1), dtype=np.float32),
    )
    model = _CaptureNeighborModel()
    with pytest.raises(ValueError, match="produced no batches"):
        train_one_epoch(
            model=model,
            trainloader=DataLoader(dataset, batch_size=1),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            criterion=build_criterion("regression"),
            device=torch.device("cpu"),
            task="regression",
        )


@pytest.mark.parametrize("n_neighbors", [3, 4, 12])
def test_standard_nichetrans_supports_odd_and_even_neighbors(n_neighbors):
    model = NicheTrans(source_length=3, target_length=2)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(4, 3), torch.randn(4, n_neighbors, 3))
    assert output.shape == (4, 2)


def test_h5mu_one_epoch_smoke_saves_and_restores_checkpoint(tmp_path):
    train_path = _write_h5mu(
        tmp_path / "train.h5mu",
        global_obs=[f"train_{index}" for index in range(4)],
    )
    test_path = _write_h5mu(
        tmp_path / "test.h5mu",
        global_obs=[f"test_{index}" for index in range(4)],
    )
    manager = H5MuDataManager(
        train_path=train_path,
        test_path=test_path,
        n_neighbors=2,
        preprocess="auto",
    )
    try:
        loader_args = SimpleNamespace(train_batch=2, test_batch=2, workers=0)
        trainloader, validationloader = h5mu_dataloader(loader_args, manager)
        set_seed(7)
        model = NicheTrans(
            source_length=manager.source_length,
            target_length=manager.target_length,
            noise_rate=0.0,
            dropout_rate=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_path = tmp_path / "best.pth"
        result = fit(
            model=model,
            trainloader=trainloader,
            validationloader=validationloader,
            optimizer=optimizer,
            criterion=build_criterion("regression"),
            device=torch.device("cpu"),
            task="regression",
            max_epochs=1,
            eval_step=1,
            checkpoint_path=checkpoint_path,
            target_panel=manager.target_panel,
            checkpoint_metadata={"dataset": "synthetic"},
            neighbor_mask_probability=0.0,
            verbose=False,
        )
        assert result["best_epoch"] == 1
        assert checkpoint_path.is_file()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert checkpoint["task"] == "regression"
        assert checkpoint["metadata"] == {"dataset": "synthetic"}
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "validation_summary" in checkpoint
    finally:
        manager.close()


def test_tutorial_notebook_is_valid_and_has_no_saved_outputs():
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "Tutorial_10.1__Train_NicheTrans_on_H5MU_data.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert "H5MuDataManager" in all_source
    assert "infer_task_type" in all_source
    assert "fit(" in all_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
