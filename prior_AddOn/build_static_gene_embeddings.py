from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .gene_embedding_loader import DEFAULT_ROOT, load_static_gene_prior
except ImportError:  # pragma: no cover - supports direct script execution.
    from gene_embedding_loader import DEFAULT_ROOT, load_static_gene_prior


SCGPT_HF_REPO = "perturblab/scgpt-continual-pretrained"
SCGPT_FILES = ("args.json", "best_model.pt", "vocab.json")

GENEFORMER_REPO = "ctheodoris/Geneformer"
GENEFORMER_VERSION = "Geneformer-V2-316M"
GENEFORMER_FILES = (
    "Geneformer-V2-316M/config.json",
    "Geneformer-V2-316M/generation_config.json",
    "Geneformer-V2-316M/model.safetensors",
    "geneformer/token_dictionary_gc104M.pkl",
    "geneformer/gene_name_id_dict_gc104M.pkl",
    "geneformer/ensembl_mapping_dict_gc104M.pkl",
)


@dataclass(frozen=True)
class ExportResult:
    model: str
    output_path: Path
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export static scGPT/Geneformer gene embeddings.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Artifact root directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["scgpt", "geneformer"],
        choices=["scgpt", "geneformer"],
        help="Models to download/export.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Use existing raw files only.")
    parser.add_argument("--force-download", action="store_true", help="Re-download files from Hugging Face.")
    parser.add_argument(
        "--scgpt-local-dir",
        type=Path,
        default=None,
        help="Optional local directory with official scGPT files: best_model.pt, vocab.json, args.json.",
    )
    parser.add_argument("--source-panel-file", type=Path, default=None, help="Optional file of genes to align.")
    parser.add_argument("--source-column", type=str, default=None, help="CSV/TSV column name for source genes.")
    parser.add_argument("--species", type=str, default=None, help="human or mouse; required with --source-panel-file.")
    parser.add_argument("--dataset-key", type=str, default=None, help="Dataset key for aligned output filenames.")
    parser.add_argument("--no-network", action="store_true", help="Disable Ensembl calls during optional alignment.")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_requested_files(root, args.models, force_download=args.force_download)

    results: list[ExportResult] = []
    if "scgpt" in args.models:
        results.append(export_scgpt_static(root, scgpt_local_dir=args.scgpt_local_dir))
    if "geneformer" in args.models:
        results.append(export_geneformer_static(root))

    manifest = update_manifest(root, results)

    if args.source_panel_file is not None:
        if not args.species or not args.dataset_key:
            raise ValueError("--species and --dataset-key are required with --source-panel-file")
        source_panel = read_source_panel(args.source_panel_file, source_column=args.source_column)
        aligned = load_static_gene_prior(
            source_panel,
            args.species,
            models=tuple(args.models),
            root=root,
            dataset_key=args.dataset_key,
            write_aligned=True,
            allow_network=not args.no_network,
        )
        manifest.setdefault("aligned_outputs", {})[args.dataset_key] = {
            model: output["coverage"] for model, output in aligned.items()
        }
        write_manifest(root, manifest)

    print(f"Wrote manifest: {root / 'manifest.json'}")


def download_requested_files(root: Path, models: list[str], force_download: bool = False) -> None:
    if "scgpt" in models:
        download_hf_files(
            repo_id=SCGPT_HF_REPO,
            filenames=SCGPT_FILES,
            local_dir=_scgpt_raw_dir(root),
            force_download=force_download,
        )
    if "geneformer" in models:
        download_hf_files(
            repo_id=GENEFORMER_REPO,
            filenames=GENEFORMER_FILES,
            local_dir=_geneformer_raw_dir(root),
            force_download=force_download,
        )


def download_hf_files(repo_id: str, filenames: tuple[str, ...], local_dir: Path, force_download: bool = False) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for downloads") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print(f"Downloading {repo_id}:{filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=local_dir,
            force_download=force_download,
        )


def export_scgpt_static(root: Path, scgpt_local_dir: Path | None = None) -> ExportResult:
    torch = _require_torch()
    import torch.nn.functional as F

    source_dir = scgpt_local_dir.resolve() if scgpt_local_dir is not None else _scgpt_raw_dir(root)
    checkpoint_path = source_dir / "best_model.pt"
    vocab_path = source_dir / "vocab.json"
    args_path = source_dir / "args.json"
    _require_files([checkpoint_path, vocab_path])

    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    embedding_weight = _state_tensor(
        state_dict,
        [
            "encoder.embedding.weight",
            "encoder.emb.weight",
            "gene_encoder.embedding.weight",
        ],
    ).detach().cpu().float()

    norm_weight = _optional_state_tensor(
        state_dict,
        ["encoder.enc_norm.weight", "encoder.norm.weight", "gene_encoder.enc_norm.weight"],
    )
    norm_bias = _optional_state_tensor(
        state_dict,
        ["encoder.enc_norm.bias", "encoder.norm.bias", "gene_encoder.enc_norm.bias"],
    )
    if norm_weight is not None and norm_bias is not None:
        embeddings_by_token = F.layer_norm(
            embedding_weight,
            normalized_shape=(embedding_weight.shape[1],),
            weight=norm_weight.detach().cpu().float(),
            bias=norm_bias.detach().cpu().float(),
        )
        encoder_norm_applied = True
    else:
        embeddings_by_token = embedding_weight
        encoder_norm_applied = False

    rows: list[tuple[str, int]] = []
    for gene, token_id in sorted(vocab.items(), key=lambda item: int(item[1])):
        if _is_special_token(gene):
            continue
        token_id = int(token_id)
        if 0 <= token_id < embeddings_by_token.shape[0]:
            rows.append((str(gene), token_id))

    token_indices = torch.tensor([token_id for _, token_id in rows], dtype=torch.long)
    embeddings = embeddings_by_token.index_select(0, token_indices).contiguous()
    gene_to_idx = {gene: idx for idx, (gene, _) in enumerate(rows)}
    token_ids = {gene: token_id for gene, token_id in rows}

    processed_dir = _processed_dir(root)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "scgpt_static.pt"
    metadata = {
        "model": "scgpt",
        "source_repo": SCGPT_HF_REPO if scgpt_local_dir is None else "",
        "source_type": "huggingface_programmatic_fallback" if scgpt_local_dir is None else "local_official_or_user_supplied",
        "source_note": (
            "Official scGPT whole-human weights are commonly distributed outside direct HF downloads; "
            "pass --scgpt-local-dir to export a manually downloaded official directory."
        ),
        "raw_dir": str(source_dir),
        "checkpoint": str(checkpoint_path),
        "vocab": str(vocab_path),
        "args": str(args_path) if args_path.exists() else "",
        "embedding_dim": int(embeddings.shape[1]),
        "n_genes": int(embeddings.shape[0]),
        "vocab_size": int(len(vocab)),
        "encoder_norm_applied": encoder_norm_applied,
    }
    torch.save(
        {
            **metadata,
            "embeddings": embeddings,
            "genes": [gene for gene, _ in rows],
            "gene_to_idx": gene_to_idx,
            "token_ids": token_ids,
            "special_tokens": [gene for gene in vocab if _is_special_token(gene)],
        },
        out_path,
    )
    print(f"Exported scGPT static embeddings: {out_path} {tuple(embeddings.shape)}")
    return ExportResult("scgpt", out_path, metadata)


def export_geneformer_static(root: Path) -> ExportResult:
    torch = _require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required to export Geneformer embeddings") from exc

    source_dir = _geneformer_raw_dir(root)
    model_path = source_dir / "Geneformer-V2-316M" / "model.safetensors"
    config_path = source_dir / "Geneformer-V2-316M" / "config.json"
    token_dict_path = source_dir / "geneformer" / "token_dictionary_gc104M.pkl"
    gene_name_path = source_dir / "geneformer" / "gene_name_id_dict_gc104M.pkl"
    ensembl_mapping_path = source_dir / "geneformer" / "ensembl_mapping_dict_gc104M.pkl"
    _require_files([model_path, config_path, token_dict_path, gene_name_path, ensembl_mapping_path])

    token_dict = _load_pickle(token_dict_path)
    gene_name_to_ensembl = _upper_ensembl_values(_load_pickle(gene_name_path))
    ensembl_mapping = _upper_ensembl_values(_load_pickle(ensembl_mapping_path))

    state_dict = load_file(str(model_path), device="cpu")
    embedding_weight = _state_tensor(
        state_dict,
        [
            "bert.embeddings.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
            "word_embeddings.weight",
        ],
    ).detach().cpu().float()

    rows: list[tuple[str, int]] = []
    for gene_id, token_id in sorted(token_dict.items(), key=lambda item: int(item[1])):
        gene_id = str(gene_id)
        token_id = int(token_id)
        if _is_special_token(gene_id) or not gene_id.upper().startswith("ENSG"):
            continue
        if 0 <= token_id < embedding_weight.shape[0]:
            rows.append((gene_id.upper(), token_id))

    token_indices = torch.tensor([token_id for _, token_id in rows], dtype=torch.long)
    embeddings = embedding_weight.index_select(0, token_indices).contiguous()
    ensembl_to_idx = {gene_id: idx for idx, (gene_id, _) in enumerate(rows)}
    token_ids = {gene_id: token_id for gene_id, token_id in rows}
    ensembl_to_gene_names = _build_ensembl_to_gene_names(gene_name_to_ensembl, ensembl_mapping)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    processed_dir = _processed_dir(root)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "geneformer_v2_316m_static.pt"
    metadata = {
        "model": "geneformer",
        "model_version": GENEFORMER_VERSION,
        "source_repo": GENEFORMER_REPO,
        "raw_dir": str(source_dir),
        "checkpoint": str(model_path),
        "config": str(config_path),
        "embedding_dim": int(embeddings.shape[1]),
        "n_genes": int(embeddings.shape[0]),
        "token_vocab_size": int(len(token_dict)),
        "model_vocab_size": int(config.get("vocab_size", 0)),
    }
    torch.save(
        {
            **metadata,
            "embeddings": embeddings,
            "genes": [gene_id for gene_id, _ in rows],
            "ensembl_to_idx": ensembl_to_idx,
            "token_ids": token_ids,
            "special_tokens": [gene_id for gene_id in token_dict if _is_special_token(str(gene_id))],
            "gene_name_to_ensembl": gene_name_to_ensembl,
            "ensembl_mapping": ensembl_mapping,
            "ensembl_to_gene_names": ensembl_to_gene_names,
        },
        out_path,
    )
    print(f"Exported Geneformer static embeddings: {out_path} {tuple(embeddings.shape)}")
    return ExportResult("geneformer", out_path, metadata)


def read_source_panel(path: Path, source_column: str | None = None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict) and "source_panel" in data:
            return [str(item) for item in data["source_panel"]]
        raise ValueError("JSON source panel must be a list or contain a source_panel key")

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if source_column is None:
                source_column = reader.fieldnames[0] if reader.fieldnames else None
            if not source_column:
                raise ValueError("Could not infer source gene column")
            return [row[source_column] for row in reader if row.get(source_column)]

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def update_manifest(root: Path, results: list[ExportResult]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {}

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.setdefault("models", {})
    for result in results:
        manifest["models"][result.model] = {
            **result.metadata,
            "output_path": str(result.output_path),
        }
    write_manifest(root, manifest)
    return manifest


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _strip_module_prefix(checkpoint[key])
        if all(hasattr(value, "shape") for value in checkpoint.values() if value is not None):
            return _strip_module_prefix(checkpoint)
    raise ValueError("Could not find a state_dict in the checkpoint")


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    stripped = {}
    for key, value in state_dict.items():
        new_key = str(key)
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]
        stripped[new_key] = value
    return stripped


def _state_tensor(state_dict: dict[str, Any], candidate_keys: list[str]) -> Any:
    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key]
    suffix_matches = [
        (key, value)
        for key, value in state_dict.items()
        if any(str(key).endswith(candidate) for candidate in candidate_keys)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0][1]
    available = ", ".join(sorted(str(key) for key in state_dict.keys())[:20])
    raise KeyError(f"Missing expected tensor keys {candidate_keys}; first available keys: {available}")


def _optional_state_tensor(state_dict: dict[str, Any], candidate_keys: list[str]) -> Any | None:
    try:
        return _state_tensor(state_dict, candidate_keys)
    except KeyError:
        return None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _upper_ensembl_values(mapping: dict[Any, Any]) -> dict[str, str]:
    return {str(key): str(value).split(".")[0].upper() for key, value in mapping.items() if value}


def _build_ensembl_to_gene_names(
    gene_name_to_ensembl: dict[str, str],
    ensembl_mapping: dict[str, str],
) -> dict[str, list[str]]:
    inverse: dict[str, list[str]] = {}
    for name, ensembl_id in gene_name_to_ensembl.items():
        inverse.setdefault(ensembl_id, [])
        if name not in inverse[ensembl_id]:
            inverse[ensembl_id].append(name)
    for name, ensembl_id in ensembl_mapping.items():
        inverse.setdefault(ensembl_id, [])
        if name not in inverse[ensembl_id]:
            inverse[ensembl_id].append(name)
    return inverse


def _is_special_token(token: str) -> bool:
    return str(token).startswith("<") and str(token).endswith(">")


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _processed_dir(root: Path) -> Path:
    return root / "processed"


def _scgpt_raw_dir(root: Path) -> Path:
    return root / "raw" / "scgpt" / SCGPT_HF_REPO.replace("/", "__")


def _geneformer_raw_dir(root: Path) -> Path:
    return root / "raw" / "geneformer" / GENEFORMER_REPO.replace("/", "__")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to export static gene embeddings") from exc
    return torch


if __name__ == "__main__":
    main()
