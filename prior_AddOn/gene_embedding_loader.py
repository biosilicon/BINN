from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(__file__).resolve().parent / "gene_embeddings"
PROCESSED_DIRNAME = "processed"

MODEL_FILE_NAMES = {
    "scgpt": "scgpt_static.pt",
    "geneformer": "geneformer_v2_316m_static.pt",
}

ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+$", re.IGNORECASE)
HUMAN_ENSEMBL_RE = re.compile(r"^ENSG\d+$", re.IGNORECASE)
MOUSE_ENSEMBL_RE = re.compile(r"^ENSMUSG\d+$", re.IGNORECASE)
ENSEMBL_VERSION_RE = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+$", re.IGNORECASE)
PEAK_RE = re.compile(r"^(chr)?[0-9XYM]+[:_-]\d+[-_]\d+$", re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"^<.*>$")


def load_static_gene_prior(
    source_panel: Iterable[Any],
    species: str,
    models: tuple[str, ...] | list[str] = ("scgpt", "geneformer"),
    root: str | Path | None = None,
    dataset_key: str | None = None,
    write_aligned: bool = False,
    allow_network: bool = True,
    use_aligned_cache: bool = True,
    write_aligned_cache: bool = True,
    precache_orthologs: bool = True,
    retry_failed_mappings: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load static priors aligned to the exact order of source_panel.

    Parameters beyond the first three are optional so existing callers can use
    the short planned signature while preparation scripts can persist artifacts.
    """

    torch = _require_torch()
    source_genes = [str(g) for g in source_panel]
    species_key = _normalize_species(species)
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    processed_dir = root_path / PROCESSED_DIRNAME

    requested_models = tuple(_normalize_model_name(m) for m in models)
    results: dict[str, dict[str, Any]] = {}
    pending_models: list[str] = []
    cache_paths: dict[str, Path] = {}
    for model_name in requested_models:
        cache_path = _aligned_cache_path(processed_dir, dataset_key, species_key, source_genes, model_name)
        cache_paths[model_name] = cache_path
        cached = _load_aligned_cache(cache_path, model_name, species_key, source_genes) if use_aligned_cache else None
        if cached is None:
            pending_models.append(model_name)
        else:
            results[model_name] = cached

    if not pending_models:
        return {model_name: results[model_name] for model_name in requested_models}

    resolver = GeneResolver(
        root_path,
        allow_network=allow_network,
        retry_failed_mappings=retry_failed_mappings,
    )

    loaded: dict[str, dict[str, Any]] = {}
    for model_name in pending_models:
        loaded[model_name] = _load_static_model(model_name, processed_dir)

    geneformer_dicts = _load_geneformer_dictionaries(processed_dir, loaded)

    if species_key == "mouse" and precache_orthologs:
        resolver.resolve_mouse_panel(source_genes, geneformer_dicts)
        resolver.retry_failed_mappings = False

    for model_name in pending_models:
        model_data = loaded[model_name]
        embedding_bank = model_data["embeddings"].detach().cpu()
        dim = int(embedding_bank.shape[1])
        aligned = torch.zeros((len(source_genes), dim), dtype=embedding_bank.dtype)
        found_mask = torch.zeros((len(source_genes),), dtype=torch.bool)
        mapping_rows: list[dict[str, Any]] = []
        seen_keys: dict[str, int] = {}

        for row_idx, raw_gene in enumerate(source_genes):
            row = _map_one_gene(
                raw_gene=raw_gene,
                species=species_key,
                model_name=model_name,
                model_data=model_data,
                geneformer_dicts=geneformer_dicts,
                resolver=resolver,
            )
            row["row_index"] = row_idx
            row["model"] = model_name
            duplicate_key = row.get("mapped_id") or row.get("normalized_input")
            if duplicate_key:
                previous = seen_keys.get(str(duplicate_key))
                if previous is not None:
                    row["duplicate_of"] = previous
                    row["reason"] = _append_reason(row.get("reason", ""), "duplicate_source_feature")
                else:
                    seen_keys[str(duplicate_key)] = row_idx

            emb_idx = row.get("embedding_index")
            if row.get("status") == "mapped" and emb_idx is not None:
                aligned[row_idx] = embedding_bank[int(emb_idx)]
                found_mask[row_idx] = True
            mapping_rows.append(row)

        coverage = _coverage(found_mask, mapping_rows, model_name, species_key)
        results[model_name] = {
            "embeddings": aligned,
            "found_mask": found_mask,
            "mapping_table": mapping_rows,
            "coverage": coverage,
        }

        if write_aligned or write_aligned_cache:
            _save_aligned_cache(
                cache_paths[model_name],
                model_name=model_name,
                dataset_key=dataset_key,
                species=species_key,
                source_panel=source_genes,
                payload=results[model_name],
            )

    resolver.flush()
    return {model_name: results[model_name] for model_name in requested_models}


def _aligned_cache_path(
    processed_dir: Path,
    dataset_key: str | None,
    species: str,
    source_panel: list[str],
    model_name: str,
) -> Path:
    aligned_dir = processed_dir / "aligned"
    if dataset_key:
        prefix = _safe_cache_component(dataset_key)
    else:
        prefix = f"auto_{species}_{_source_panel_hash(species, source_panel)}"
    return aligned_dir / f"{prefix}_{model_name}.pt"


def _load_aligned_cache(
    path: Path,
    model_name: str,
    species: str,
    source_panel: list[str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    torch = _require_torch()
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if data.get("model") != model_name or data.get("species") != species:
        return None
    if [str(g) for g in data.get("source_panel", [])] != source_panel:
        return None
    required_keys = {"embeddings", "found_mask", "mapping_table", "coverage"}
    if not required_keys.issubset(data):
        return None
    if int(data["embeddings"].shape[0]) != len(source_panel):
        return None
    if int(data["found_mask"].shape[0]) != len(source_panel):
        return None
    return {
        "embeddings": data["embeddings"],
        "found_mask": data["found_mask"],
        "mapping_table": data["mapping_table"],
        "coverage": data["coverage"],
    }


def _save_aligned_cache(
    path: Path,
    model_name: str,
    dataset_key: str | None,
    species: str,
    source_panel: list[str],
    payload: dict[str, Any],
) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "dataset_key": dataset_key,
            "species": species,
            "source_panel": source_panel,
            "source_panel_hash": _source_panel_hash(species, source_panel),
            "embeddings": payload["embeddings"],
            "found_mask": payload["found_mask"],
            "mapping_table": payload["mapping_table"],
            "coverage": payload["coverage"],
        },
        path,
    )


def _source_panel_hash(species: str, source_panel: list[str]) -> str:
    payload = json.dumps(
        {"species": species, "source_panel": source_panel},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_cache_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "dataset"


def _map_one_gene(
    raw_gene: str,
    species: str,
    model_name: str,
    model_data: dict[str, Any],
    geneformer_dicts: dict[str, Any],
    resolver: "GeneResolver",
) -> dict[str, Any]:
    normalized = normalize_gene_name(raw_gene)
    base_row = {
        "input_gene": raw_gene,
        "normalized_input": normalized,
        "species": species,
        "mapped_id": "",
        "mapped_symbol": "",
        "token_id": "",
        "embedding_index": None,
    }

    if not normalized:
        return {
            **base_row,
            "status": "invalid_input",
            "reason": "empty_gene_name",
        }
    if is_non_gene_feature(normalized):
        return {
            **base_row,
            "status": "non_gene_feature",
            "reason": "looks_like_genomic_interval_or_special_token",
        }

    if model_name == "scgpt":
        return _map_scgpt(normalized, species, model_data, geneformer_dicts, resolver, base_row)
    if model_name == "geneformer":
        return _map_geneformer(normalized, species, model_data, geneformer_dicts, resolver, base_row)
    raise ValueError(f"Unsupported model: {model_name}")


def _map_scgpt(
    gene: str,
    species: str,
    model_data: dict[str, Any],
    geneformer_dicts: dict[str, Any],
    resolver: "GeneResolver",
    base_row: dict[str, Any],
) -> dict[str, Any]:
    gene_to_idx = model_data.get("gene_to_idx", {})
    token_ids = model_data.get("token_ids", {})

    if species == "human":
        symbol = _match_symbol(gene, gene_to_idx)
        if symbol is not None:
            return _mapped_row(base_row, symbol, symbol, token_ids.get(symbol), gene_to_idx[symbol])

        if is_human_ensembl(gene):
            return _map_scgpt_from_human_ensembl(gene, model_data, geneformer_dicts, base_row)

        human_ensembl = _symbol_to_human_ensembl(gene, geneformer_dicts)
        if human_ensembl:
            row = _map_scgpt_from_human_ensembl(human_ensembl, model_data, geneformer_dicts, base_row)
            if row["status"] == "mapped":
                return row
        return {
            **base_row,
            "status": "missing_embedding",
            "reason": "human_symbol_not_in_scgpt_vocab",
        }

    mouse_resolution = resolver.resolve_mouse_gene(gene, geneformer_dicts)
    if mouse_resolution.status != "mapped":
        return {
            **base_row,
            "status": mouse_resolution.status,
            "reason": mouse_resolution.reason,
        }
    row = _map_scgpt_from_human_ensembl(
        mouse_resolution.human_ensembl,
        model_data,
        geneformer_dicts,
        base_row,
    )
    row["reason"] = _append_reason(row.get("reason", ""), "mouse_to_human_ortholog")
    return row


def _map_geneformer(
    gene: str,
    species: str,
    model_data: dict[str, Any],
    geneformer_dicts: dict[str, Any],
    resolver: "GeneResolver",
    base_row: dict[str, Any],
) -> dict[str, Any]:
    ensembl_to_idx = model_data.get("ensembl_to_idx", {})
    token_ids = model_data.get("token_ids", {})

    if species == "human":
        if is_human_ensembl(gene):
            human_ensembl = gene.upper()
        elif is_mouse_ensembl(gene):
            mouse_resolution = resolver.resolve_mouse_gene(gene, geneformer_dicts)
            if mouse_resolution.status != "mapped":
                return {
                    **base_row,
                    "status": mouse_resolution.status,
                    "reason": mouse_resolution.reason,
                }
            human_ensembl = mouse_resolution.human_ensembl
        else:
            human_ensembl = _symbol_to_human_ensembl(gene, geneformer_dicts)
            if not human_ensembl:
                return {
                    **base_row,
                    "status": "unmapped",
                    "reason": "human_symbol_not_in_geneformer_dictionaries",
                }
    else:
        mouse_resolution = resolver.resolve_mouse_gene(gene, geneformer_dicts)
        if mouse_resolution.status != "mapped":
            return {
                **base_row,
                "status": mouse_resolution.status,
                "reason": mouse_resolution.reason,
            }
        human_ensembl = mouse_resolution.human_ensembl

    human_ensembl = human_ensembl.upper()
    if human_ensembl not in ensembl_to_idx:
        return {
            **base_row,
            "mapped_id": human_ensembl,
            "status": "missing_embedding",
            "reason": "human_ensembl_not_in_geneformer_vocab",
        }
    symbol = _first_symbol_for_ensembl(human_ensembl, geneformer_dicts)
    return _mapped_row(
        base_row,
        human_ensembl,
        symbol,
        token_ids.get(human_ensembl),
        ensembl_to_idx[human_ensembl],
    )


def _map_scgpt_from_human_ensembl(
    human_ensembl: str,
    model_data: dict[str, Any],
    geneformer_dicts: dict[str, Any],
    base_row: dict[str, Any],
) -> dict[str, Any]:
    gene_to_idx = model_data.get("gene_to_idx", {})
    token_ids = model_data.get("token_ids", {})
    candidates = geneformer_dicts.get("ensembl_to_gene_names", {}).get(human_ensembl.upper(), [])
    matched = []
    for candidate in candidates:
        symbol = _match_symbol(candidate, gene_to_idx)
        if symbol is not None and symbol not in matched:
            matched.append(symbol)

    if len(matched) == 1:
        symbol = matched[0]
        return _mapped_row(base_row, human_ensembl.upper(), symbol, token_ids.get(symbol), gene_to_idx[symbol])
    if len(matched) > 1:
        return {
            **base_row,
            "mapped_id": human_ensembl.upper(),
            "status": "ambiguous",
            "reason": "multiple_scgpt_symbols_for_human_ensembl",
            "candidate_symbols": "|".join(matched),
        }
    return {
        **base_row,
        "mapped_id": human_ensembl.upper(),
        "status": "missing_embedding",
        "reason": "human_ensembl_symbol_not_in_scgpt_vocab",
        "candidate_symbols": "|".join(candidates),
    }


def _mapped_row(
    base_row: dict[str, Any],
    mapped_id: str,
    mapped_symbol: str,
    token_id: Any,
    embedding_index: int,
) -> dict[str, Any]:
    return {
        **base_row,
        "mapped_id": mapped_id,
        "mapped_symbol": mapped_symbol or "",
        "token_id": "" if token_id is None else token_id,
        "embedding_index": int(embedding_index),
        "status": "mapped",
        "reason": "",
    }


def normalize_gene_name(gene: Any) -> str:
    value = "" if gene is None else str(gene).strip()
    match = ENSEMBL_VERSION_RE.match(value)
    if match:
        return match.group(1).upper()
    if ENSEMBL_RE.match(value):
        return value.upper()
    return value


def is_non_gene_feature(gene: str) -> bool:
    return bool(SPECIAL_TOKEN_RE.match(gene) or PEAK_RE.match(gene))


def is_human_ensembl(gene: str) -> bool:
    return bool(HUMAN_ENSEMBL_RE.match(gene))


def is_mouse_ensembl(gene: str) -> bool:
    return bool(MOUSE_ENSEMBL_RE.match(gene))


def _symbol_to_human_ensembl(gene: str, geneformer_dicts: dict[str, Any]) -> str | None:
    exact = geneformer_dicts.get("gene_name_to_ensembl", {}).get(gene)
    if exact:
        return str(exact).upper()
    upper = gene.upper()
    upper_exact = geneformer_dicts.get("gene_name_to_ensembl", {}).get(upper)
    if upper_exact:
        return str(upper_exact).upper()
    alias = geneformer_dicts.get("ensembl_mapping", {}).get(upper)
    if alias:
        return str(alias).upper()
    return None


def _first_symbol_for_ensembl(human_ensembl: str, geneformer_dicts: dict[str, Any]) -> str:
    candidates = geneformer_dicts.get("ensembl_to_gene_names", {}).get(human_ensembl.upper(), [])
    return candidates[0] if candidates else ""


def _match_symbol(gene: str, gene_to_idx: dict[str, int]) -> str | None:
    if gene in gene_to_idx:
        return gene
    upper = gene.upper()
    if upper in gene_to_idx:
        return upper
    return None


@dataclass
class Resolution:
    status: str
    human_ensembl: str = ""
    human_symbol: str = ""
    reason: str = ""


class GeneResolver:
    """Resolve species-specific gene identifiers to human Ensembl ids."""

    cache_fields = [
        "source_species",
        "input_gene",
        "normalized_gene",
        "human_ensembl",
        "human_symbol",
        "status",
        "reason",
        "updated_at",
    ]

    def __init__(
        self,
        root: Path,
        allow_network: bool = True,
        timeout: int = 20,
        retry_failed_mappings: bool = False,
    ):
        self.root = root
        self.allow_network = allow_network
        self.timeout = timeout
        self.retry_failed_mappings = retry_failed_mappings
        self.cache_path = root / PROCESSED_DIRNAME / "mapping_cache.tsv"
        self._cache: dict[tuple[str, str], dict[str, str]] = {}
        self._dirty = False
        self._load_cache()

    def resolve_mouse_panel(self, genes: Iterable[Any], geneformer_dicts: dict[str, Any]) -> None:
        seen: set[str] = set()
        for gene in genes:
            normalized = normalize_gene_name(gene)
            if normalized in seen:
                continue
            seen.add(normalized)

            if not normalized:
                resolution = Resolution("invalid_input", reason="empty_gene_name")
                self._update_cache("mouse", str(gene), normalized, resolution)
                continue
            if is_non_gene_feature(normalized):
                resolution = Resolution("non_gene_feature", reason="looks_like_genomic_interval_or_special_token")
                self._update_cache("mouse", str(gene), normalized, resolution)
                continue

            self.resolve_mouse_gene(str(gene), geneformer_dicts)

    def resolve_mouse_gene(self, gene: str, geneformer_dicts: dict[str, Any]) -> Resolution:
        normalized = normalize_gene_name(gene)
        cache_key = ("mouse", normalized)
        cached = self._cache.get(cache_key)
        if cached and not self._should_retry_cached_mapping(cached):
            return Resolution(
                status=cached.get("status", "unmapped"),
                human_ensembl=cached.get("human_ensembl", ""),
                human_symbol=cached.get("human_symbol", ""),
                reason=cached.get("reason", ""),
            )

        if not self.allow_network:
            resolution = Resolution("unmapped", reason="mouse_mapping_not_in_cache")
            self._update_cache("mouse", gene, normalized, resolution)
            return resolution

        try:
            source_id = normalized if is_mouse_ensembl(normalized) else self._lookup_symbol("mus_musculus", normalized)
            if not source_id:
                resolution = Resolution("unmapped", reason="ensembl_mouse_symbol_lookup_failed")
                self._update_cache("mouse", gene, normalized, resolution)
                return resolution

            homologies = self._fetch_homologies("mus_musculus", source_id)
            one_to_one = [
                h for h in homologies
                if h.get("target", {}).get("species") == "homo_sapiens"
                and h.get("type") == "ortholog_one2one"
                and h.get("target", {}).get("id")
            ]
            if len(one_to_one) != 1:
                status = "ambiguous" if len(one_to_one) > 1 else "unmapped"
                reason = "multiple_human_one_to_one_orthologs" if len(one_to_one) > 1 else "no_human_one_to_one_ortholog"
                resolution = Resolution(status, reason=reason)
                self._update_cache("mouse", gene, normalized, resolution)
                return resolution

            human_ensembl = str(one_to_one[0]["target"]["id"]).upper()
            human_symbol = _first_symbol_for_ensembl(human_ensembl, geneformer_dicts)
            resolution = Resolution("mapped", human_ensembl=human_ensembl, human_symbol=human_symbol)
            self._update_cache("mouse", gene, normalized, resolution)
            return resolution
        except Exception as exc:  # network errors should be traceable, not fatal to a full panel.
            resolution = Resolution("unmapped", reason=f"ensembl_request_failed:{type(exc).__name__}")
            self._update_cache("mouse", gene, normalized, resolution)
            return resolution

    def flush(self) -> None:
        if not self._dirty:
            return
        self._write_cache()
        self._dirty = False

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._cache.values(), key=lambda r: (r["source_species"], r["normalized_gene"]))
        tmp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.cache_fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(self.cache_path)

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                key = (row.get("source_species", ""), row.get("normalized_gene", ""))
                if all(key):
                    self._cache[key] = {field: row.get(field, "") for field in self.cache_fields}

    def _update_cache(self, species: str, input_gene: str, normalized: str, resolution: Resolution) -> None:
        self._cache[(species, normalized)] = {
            "source_species": species,
            "input_gene": input_gene,
            "normalized_gene": normalized,
            "human_ensembl": resolution.human_ensembl,
            "human_symbol": resolution.human_symbol,
            "status": resolution.status,
            "reason": resolution.reason,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._dirty = True
        self.flush()

    def _should_retry_cached_mapping(self, cached: dict[str, str]) -> bool:
        if not self.allow_network:
            return False
        if not self.retry_failed_mappings:
            return False
        return str(cached.get("reason", "")).startswith("ensembl_request_failed:")

    def _lookup_symbol(self, species: str, symbol: str) -> str | None:
        requests = _require_requests()
        url = f"https://rest.ensembl.org/lookup/symbol/{species}/{symbol}"
        response = requests.get(
            url,
            params={"content-type": "application/json"},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        gene_id = data.get("id")
        return str(gene_id).upper() if gene_id else None

    def _fetch_homologies(self, species: str, ensembl_gene_id: str) -> list[dict[str, Any]]:
        requests = _require_requests()
        url = f"https://rest.ensembl.org/homology/id/{species}/{ensembl_gene_id}"
        response = requests.get(
            url,
            params={
                "target_species": "homo_sapiens",
                "type": "orthologues",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        rows = data.get("data", [])
        if not rows:
            return []
        return rows[0].get("homologies", [])


def _load_static_model(model_name: str, processed_dir: Path) -> dict[str, Any]:
    torch = _require_torch()
    path = processed_dir / MODEL_FILE_NAMES[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {model_name} static embedding file: {path}. "
            "Run prior_AddOn/build_static_gene_embeddings.py first."
        )
    data = torch.load(path, map_location="cpu")
    if "embeddings" not in data:
        raise ValueError(f"{path} does not contain an 'embeddings' tensor")
    return data


def _load_geneformer_dictionaries(
    processed_dir: Path,
    loaded: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if "geneformer" in loaded:
        source = loaded["geneformer"]
    else:
        path = processed_dir / MODEL_FILE_NAMES["geneformer"]
        source = _load_static_model("geneformer", processed_dir) if path.exists() else {}

    return {
        "gene_name_to_ensembl": source.get("gene_name_to_ensembl", {}),
        "ensembl_mapping": source.get("ensembl_mapping", {}),
        "ensembl_to_gene_names": source.get("ensembl_to_gene_names", {}),
    }


def _coverage(found_mask: Any, mapping_rows: list[dict[str, Any]], model_name: str, species: str) -> dict[str, Any]:
    total = int(len(mapping_rows))
    found = int(found_mask.sum().item()) if total else 0
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


def _normalize_model_name(model_name: str) -> str:
    key = str(model_name).lower().replace("-", "_")
    if key in {"scgpt", "sc_gpt"}:
        return "scgpt"
    if key in {"geneformer", "geneformer_v2", "geneformer_v2_316m"}:
        return "geneformer"
    raise ValueError(f"Unsupported model name: {model_name}")


def _normalize_species(species: str) -> str:
    key = str(species).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"human", "homo_sapiens", "hsapiens", "hs"}:
        return "human"
    if key in {"mouse", "mus_musculus", "mmusculus", "mm", "murine"}:
        return "mouse"
    raise ValueError("species must be human/homo_sapiens or mouse/mus_musculus")


def _append_reason(reason: str, addition: str) -> str:
    if not reason:
        return addition
    if addition in reason.split(";"):
        return reason
    return f"{reason};{addition}"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load static gene embedding tensors") from exc
    return torch


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for Ensembl mapping when allow_network=True") from exc
    return requests
