from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .gene_embedding_loader import DEFAULT_ROOT, load_static_gene_prior
except ImportError:  # pragma: no cover - supports direct script execution.
    from gene_embedding_loader import DEFAULT_ROOT, load_static_gene_prior


DEFAULT_AD_ADATA_PATH = (
    "/home1/shezixi/data/in_silico_central_dogma_challenge/unprocessed/"
    "2023_nn_AD_mouse/AD_model_adata_protein"
)
DEFAULT_TRAINING_SLIDE = "13months-disease-replicate_1_random.h5ad"
DEFAULT_TESTING_SLIDE = "13months-disease-replicate_2_random.h5ad"
DEFAULT_DATASET_KEY = None
DEFAULT_REPORT_DIRNAME = "starmap_plus_mapping"

SOURCE_PANEL_COLUMNS = ["row_index", "gene"]
MAPPING_TABLE_COLUMNS = [
    "row_index",
    "source_gene",
    "model",
    "input_gene",
    "normalized_input",
    "species",
    "mapped_id",
    "mapped_symbol",
    "token_id",
    "embedding_index",
    "status",
    "reason",
    "candidate_symbols",
    "duplicate_of",
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    embedding_root = args.embedding_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else embedding_root / "processed" / DEFAULT_REPORT_DIRNAME
    )

    source_panel = load_starmap_plus_source_panel(
        ad_adata_path=args.ad_adata_path,
        n_top_genes=args.n_top_genes,
        training_slide=args.training_slide,
        testing_slide=args.testing_slide,
    )
    print(f"Loaded STARmap PLUS source panel: {len(source_panel)} genes")

    priors = build_prior_cache(
        source_panel=source_panel,
        embedding_root=embedding_root,
        models=tuple(args.models),
        allow_network=not args.no_network,
        retry_failed_mappings=args.retry_failed_mappings,
        use_aligned_cache=not args.rebuild_aligned_cache,
    )
    write_mapping_reports(source_panel, priors, output_dir)

    for model_name, prior in priors.items():
        coverage = prior.get("coverage", {})
        print(
            f"{model_name}: mapped {coverage.get('n_found', 0)}/"
            f"{coverage.get('n_features', len(source_panel))} "
            f"({coverage.get('coverage', 0.0):.2%})"
        )
    print(f"Wrote mapping reports: {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute load_static_gene_prior aligned caches and mapping reports "
            "for Tutorial 5.1 STARmap PLUS mouse source genes."
        )
    )
    parser.add_argument(
        "--ad-adata-path",
        type=Path,
        default=Path(DEFAULT_AD_ADATA_PATH),
        help="Directory containing STARmap PLUS AD .h5ad slides.",
    )
    parser.add_argument(
        "--training-slide",
        type=str,
        default=DEFAULT_TRAINING_SLIDE,
        help="AD training slide filename.",
    )
    parser.add_argument(
        "--testing-slide",
        type=str,
        default=DEFAULT_TESTING_SLIDE,
        help="AD testing slide filename.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of HVGs to compute when a slide has no highly_variable column.",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing prior_AddOn gene embedding artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["scgpt", "geneformer"],
        choices=["scgpt", "geneformer"],
        help="Static prior models to align.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for human-readable mapping reports. Defaults to "
            "<embedding-root>/processed/starmap_plus_mapping."
        ),
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Disable Ensembl API calls and use only existing mapping_cache.tsv rows.",
    )
    parser.add_argument(
        "--retry-failed-mappings",
        action="store_true",
        help="Retry cached Ensembl request failures when network access is enabled.",
    )
    parser.add_argument(
        "--rebuild-aligned-cache",
        action="store_true",
        help="Ignore existing aligned cache files and rebuild them from static priors.",
    )
    return parser.parse_args(argv)


def load_starmap_plus_source_panel(
    ad_adata_path: str | Path,
    n_top_genes: int = 2000,
    training_slide: str = DEFAULT_TRAINING_SLIDE,
    testing_slide: str = DEFAULT_TESTING_SLIDE,
) -> list[str]:
    """Load the same unfiltered source panel used by Tutorial 5.1."""

    root = Path(ad_adata_path)
    slide_paths = [root / training_slide, root / testing_slide]
    missing = [str(path) for path in slide_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing STARmap PLUS AD slide(s):\n" + "\n".join(missing))

    adatas = [_read_h5ad(path) for path in slide_paths]
    var_names = adatas[0].var_names
    for path, adata in zip(slide_paths[1:], adatas[1:]):
        if not var_names.equals(adata.var_names):
            raise ValueError(
                "STARmap PLUS AD slides must have identical var_names in the same order. "
                f"Mismatch found in {path}."
            )

    masks = [
        _highly_variable_mask(adata, n_top_genes=n_top_genes, slide_path=path)
        for path, adata in zip(slide_paths, adatas)
    ]
    source_mask = [all(values) for values in zip(*masks)]
    return [str(gene) for gene, keep in zip(var_names, source_mask) if keep]


def build_prior_cache(
    source_panel: Iterable[Any],
    embedding_root: str | Path = DEFAULT_ROOT,
    models: tuple[str, ...] | list[str] = ("scgpt", "geneformer"),
    allow_network: bool = True,
    retry_failed_mappings: bool = False,
    use_aligned_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build auto-hash aligned caches compatible with load_static_gene_prior."""

    return load_static_gene_prior(
        source_panel=source_panel,
        species="mouse",
        models=models,
        root=embedding_root,
        dataset_key=DEFAULT_DATASET_KEY,
        allow_network=allow_network,
        write_aligned_cache=True,
        precache_orthologs=True,
        retry_failed_mappings=retry_failed_mappings,
        use_aligned_cache=use_aligned_cache,
    )


def write_mapping_reports(
    source_panel: Iterable[Any],
    priors: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Write human-readable reports for the aligned prior mapping results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_genes = [str(gene) for gene in source_panel]

    _write_tsv(
        output_path / "source_panel.tsv",
        ({"row_index": idx, "gene": gene} for idx, gene in enumerate(source_genes)),
        SOURCE_PANEL_COLUMNS,
    )

    all_rows: list[dict[str, Any]] = []
    for model_name, prior in priors.items():
        rows = _mapping_rows_for_model(model_name, prior, source_genes)
        all_rows.extend(rows)
        _write_tsv(output_path / f"mapping_table_{model_name}.tsv", rows, MAPPING_TABLE_COLUMNS)

    _write_tsv(output_path / "mapping_table_long.tsv", all_rows, MAPPING_TABLE_COLUMNS)
    _write_tsv(
        output_path / "unmapped_or_missing.tsv",
        (row for row in all_rows if row.get("status") != "mapped"),
        MAPPING_TABLE_COLUMNS,
    )

    summary = {
        "source_panel_size": len(source_genes),
        "models": {
            model_name: prior.get("coverage", {})
            for model_name, prior in priors.items()
        },
    }
    with (output_path / "coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _highly_variable_mask(adata: Any, n_top_genes: int, slide_path: Path) -> list[bool]:
    if "highly_variable" not in adata.var.columns:
        _compute_highly_variable_genes(adata, n_top_genes=n_top_genes)

    if "highly_variable" not in adata.var.columns:
        raise ValueError(f"{slide_path} does not contain or compute a highly_variable column.")

    mask = [bool(value) for value in adata.var["highly_variable"].values]
    if len(mask) != adata.n_vars:
        raise ValueError(
            f"{slide_path} has invalid highly_variable length: "
            f"{len(mask)} != {adata.n_vars}."
        )
    return mask


def _read_h5ad(path: Path) -> Any:
    return _require_scanpy().read_h5ad(path)


def _compute_highly_variable_genes(adata: Any, n_top_genes: int) -> None:
    _require_scanpy().pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )


def _require_scanpy() -> Any:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required to read STARmap PLUS .h5ad files and compute HVGs"
        ) from exc
    return sc


def _mapping_rows_for_model(
    model_name: str,
    prior: dict[str, Any],
    source_panel: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fallback_idx, raw_row in enumerate(prior.get("mapping_table", [])):
        row = dict(raw_row)
        row_index = row.get("row_index", fallback_idx)
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            row_index_int = fallback_idx

        source_gene = source_panel[row_index_int] if 0 <= row_index_int < len(source_panel) else ""
        normalized = {
            **row,
            "row_index": row_index_int,
            "source_gene": source_gene,
            "model": row.get("model") or model_name,
        }
        rows.append({column: _format_tsv_value(normalized.get(column, "")) for column in MAPPING_TABLE_COLUMNS})
    return rows


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_tsv_value(row.get(column, "")) for column in columns})


def _format_tsv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return value


if __name__ == "__main__":
    main()
