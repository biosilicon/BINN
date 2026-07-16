from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.special
import scipy.stats
import torch
from torch import nn


_TASKS = {"auto", "regression", "binary"}


def infer_task_type(
    train_value_type: str,
    test_value_type: str,
    requested: str = "auto",
) -> str:
    """Resolve regression/binary behavior from matching H5MU assay metadata."""

    train_value_type = str(train_value_type).strip().lower()
    test_value_type = str(test_value_type).strip().lower()
    requested = str(requested).strip().lower()
    if requested not in _TASKS:
        raise ValueError(f"requested task must be one of {sorted(_TASKS)}")
    if train_value_type != test_value_type:
        raise ValueError(
            "Training and testing target value_type must match, got "
            f"{train_value_type!r} and {test_value_type!r}."
        )
    if requested != "auto":
        return requested
    return "binary" if train_value_type == "binary" else "regression"


def build_criterion(task: str) -> nn.Module:
    task = _normalize_task(task)
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def resolve_device(requested: str = "auto") -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    trainloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    neighbor_mask_probability: float = 0.3,
    neighbor_keep_probability: float = 0.5,
) -> dict[str, float | int]:
    """Train for one epoch and return sample-weighted loss statistics."""

    task = _normalize_task(task)
    _validate_probability(neighbor_mask_probability, "neighbor_mask_probability")
    _validate_probability(neighbor_keep_probability, "neighbor_keep_probability")
    model.train()
    total_loss = 0.0
    n_samples = 0
    non_blocking = device.type == "cuda"

    for source, target, source_neighbors, _ in trainloader:
        source = source.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        source_neighbors = source_neighbors.to(device, non_blocking=non_blocking)
        _validate_targets(target, task)

        if neighbor_mask_probability > 0 and bool(
            torch.rand((), device=device) < neighbor_mask_probability
        ):
            keep_mask = torch.rand(
                (*source_neighbors.shape[:-1], 1), device=device
            ) < neighbor_keep_probability
            source_neighbors = source_neighbors * keep_mask

        outputs = model(source, source_neighbors)
        loss = criterion(outputs, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = source.size(0)
        total_loss += float(loss.detach()) * batch_size
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError(
            "trainloader produced no batches; reduce train_batch or disable drop_last."
        )
    return {"loss": total_loss / n_samples, "n_samples": n_samples}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    target_panel: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a model and return summary plus per-target metrics."""

    task = _normalize_task(task)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_loss = 0.0
    n_samples = 0
    non_blocking = device.type == "cuda"

    for source, target, source_neighbors, _ in dataloader:
        source = source.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        source_neighbors = source_neighbors.to(device, non_blocking=non_blocking)
        _validate_targets(target, task)

        outputs = model(source, source_neighbors)
        loss = criterion(outputs, target)
        batch_size = source.size(0)
        total_loss += float(loss) * batch_size
        n_samples += batch_size
        predictions.append(outputs.detach().cpu())
        targets.append(target.detach().cpu())

    if n_samples == 0:
        raise ValueError("validation dataloader produced no batches.")

    prediction_array = torch.cat(predictions, dim=0).numpy()
    target_array = torch.cat(targets, dim=0).numpy()
    if prediction_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("model predictions and targets must both have shape [samples, features].")
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"prediction shape {prediction_array.shape} does not match target shape "
            f"{target_array.shape}."
        )

    panel = _normalize_panel(target_panel, target_array.shape[1])
    mean_loss = total_loss / n_samples
    if task == "binary":
        return _binary_metrics(prediction_array, target_array, panel, mean_loss)
    return _regression_metrics(prediction_array, target_array, panel, mean_loss)


def fit(
    model: nn.Module,
    trainloader: Iterable,
    validationloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    max_epochs: int,
    eval_step: int,
    checkpoint_path: str | os.PathLike[str],
    *,
    scheduler: Any | None = None,
    target_panel: Sequence[Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    neighbor_mask_probability: float = 0.3,
    neighbor_keep_probability: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train with periodic validation, save, and restore the best checkpoint."""

    task = _normalize_task(task)
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if eval_step <= 0:
        raise ValueError("eval_step must be positive.")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch: int | None = None
    best_evaluation: dict[str, Any] | None = None

    for epoch_index in range(max_epochs):
        epoch = epoch_index + 1
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_result = train_one_epoch(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task=task,
            neighbor_mask_probability=neighbor_mask_probability,
            neighbor_keep_probability=neighbor_keep_probability,
        )
        should_evaluate = epoch % eval_step == 0 or epoch == max_epochs
        evaluation = None
        improved = False
        monitor_name = None
        monitor_score = None
        if should_evaluate:
            evaluation = evaluate(
                model=model,
                dataloader=validationloader,
                criterion=criterion,
                device=device,
                task=task,
                target_panel=target_panel,
            )
            monitor_name, monitor_score = _monitor_value(evaluation["summary"], task)
            improved = monitor_score > best_score
            if improved:
                best_score = monitor_score
                best_epoch = epoch
                best_evaluation = evaluation

        if scheduler is not None:
            scheduler.step()

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_result["loss"],
        }
        if evaluation is not None:
            history_row.update(
                {f"validation_{key}": value for key, value in evaluation["summary"].items()}
            )
        history.append(history_row)

        if improved:
            checkpoint = {
                "epoch": epoch,
                "task": task,
                "monitor": monitor_name,
                "monitor_score": monitor_score,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "validation_summary": evaluation["summary"],
                "validation_per_feature": evaluation["per_feature"],
                "history": list(history),
                "metadata": dict(checkpoint_metadata or {}),
            }
            _atomic_torch_save(checkpoint, checkpoint_path)

        if verbose:
            message = f"Epoch {epoch}/{max_epochs} - train_loss={train_result['loss']:.6f}"
            if evaluation is not None:
                message += f" - validation_loss={evaluation['summary']['loss']:.6f}"
                message += f" - {monitor_name}={monitor_score:.6f}"
                if improved:
                    message += " - best"
            print(message)

    if best_epoch is None or best_evaluation is None:
        raise RuntimeError("training completed without a validation result.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_summary": best_evaluation["summary"],
        "best_per_feature": best_evaluation["per_feature"],
        "checkpoint_path": str(checkpoint_path),
    }


def _regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    panel: list[str],
    loss: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(panel):
        pred = prediction[:, feature_index]
        truth = target[:, feature_index]
        finite = np.isfinite(pred) & np.isfinite(truth)
        pred = pred[finite]
        truth = truth[finite]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2))) if len(truth) else math.nan
        pearson = math.nan
        spearman = math.nan
        if len(truth) >= 2 and np.ptp(pred) > 0 and np.ptp(truth) > 0:
            pearson = float(scipy.stats.pearsonr(pred, truth).statistic)
            spearman = float(scipy.stats.spearmanr(pred, truth).statistic)
        rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "rmse": rmse,
                "n_valid": int(len(truth)),
            }
        )

    pearsons = np.asarray([row["pearson"] for row in rows], dtype=float)
    spearmans = np.asarray([row["spearman"] for row in rows], dtype=float)
    rmses = np.asarray([row["rmse"] for row in rows], dtype=float)
    summary = {
        "loss": float(loss),
        "pearson_mean": _finite_mean(pearsons),
        "spearman_mean": _finite_mean(spearmans),
        "rmse_mean": _finite_mean(rmses),
        "n_valid_pearson": int(np.isfinite(pearsons).sum()),
        "n_targets": len(rows),
    }
    return {"summary": summary, "per_feature": rows}


def _binary_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    panel: list[str],
    loss: float,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency is documented.
        raise ImportError("Binary evaluation requires scikit-learn.") from exc

    probabilities = scipy.special.expit(logits)
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(panel):
        scores = probabilities[:, feature_index]
        truth = target[:, feature_index]
        finite = np.isfinite(scores) & np.isfinite(truth)
        scores = scores[finite]
        truth = truth[finite]
        auroc = math.nan
        if len(truth) >= 2 and np.unique(truth).size == 2:
            auroc = float(roc_auc_score(truth, scores))
        rows.append(
            {
                "feature": feature,
                "auroc": auroc,
                "n_valid": int(len(truth)),
                "n_positive": int((truth == 1).sum()),
            }
        )

    aurocs = np.asarray([row["auroc"] for row in rows], dtype=float)
    summary = {
        "loss": float(loss),
        "auroc_mean": _finite_mean(aurocs),
        "n_valid_auroc": int(np.isfinite(aurocs).sum()),
        "n_targets": len(rows),
    }
    return {"summary": summary, "per_feature": rows}


def _monitor_value(summary: dict[str, Any], task: str) -> tuple[str, float]:
    primary_name = "auroc_mean" if task == "binary" else "pearson_mean"
    primary_value = float(summary[primary_name])
    if math.isfinite(primary_value):
        return primary_name, primary_value
    return "negative_validation_loss", -float(summary["loss"])


def _normalize_panel(panel: Sequence[Any] | None, n_targets: int) -> list[str]:
    if panel is None:
        return [str(index) for index in range(n_targets)]
    values = [str(value) for value in panel]
    if len(values) != n_targets:
        raise ValueError(
            f"target_panel has {len(values)} features, expected {n_targets}."
        )
    return values


def _validate_targets(target: torch.Tensor, task: str) -> None:
    if task == "binary" and not bool(torch.all((target == 0) | (target == 1))):
        raise ValueError("binary targets must contain only 0 and 1 values.")


def _normalize_task(task: str) -> str:
    task = str(task).strip().lower()
    if task not in {"regression", "binary"}:
        raise ValueError("task must be 'regression' or 'binary'.")
    return task


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _atomic_torch_save(checkpoint: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


__all__ = [
    "build_criterion",
    "evaluate",
    "fit",
    "infer_task_type",
    "resolve_device",
    "train_one_epoch",
]
