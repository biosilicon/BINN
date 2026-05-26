from __future__ import annotations

import csv
import json

import pandas as pd
import pytest
import torch

from prior_AddOn.build_starmap_plus_mouse_mapping import (
    DEFAULT_TESTING_SLIDE,
    DEFAULT_TRAINING_SLIDE,
    build_prior_cache,
    load_starmap_plus_source_panel,
    write_mapping_reports,
)
from prior_AddOn.gene_embedding_loader import load_static_gene_prior


def test_load_starmap_plus_source_panel_uses_hvg_intersection(tmp_path):
    genes = ["A", "B", "C", "D"]
    training_path = tmp_path / DEFAULT_TRAINING_SLIDE
    testing_path = tmp_path / DEFAULT_TESTING_SLIDE
    fake_adatas = {
        training_path: _FakeAdata(genes, highly_variable=[True, True, False, False]),
        testing_path: _FakeAdata(genes, highly_variable=[False, True, True, False]),
    }
    _touch_slides(fake_adatas)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "prior_AddOn.build_starmap_plus_mouse_mapping._read_h5ad",
            lambda path: fake_adatas[path],
        )
        source_panel = load_starmap_plus_source_panel(tmp_path, n_top_genes=2)

    assert source_panel == ["B"]


def test_load_starmap_plus_source_panel_computes_missing_hvg(tmp_path, monkeypatch):
    genes = ["A", "B", "C"]
    training_path = tmp_path / DEFAULT_TRAINING_SLIDE
    testing_path = tmp_path / DEFAULT_TESTING_SLIDE
    fake_adatas = {
        training_path: _FakeAdata(genes, slide_label="train"),
        testing_path: _FakeAdata(genes, slide_label="test"),
    }
    _touch_slides(fake_adatas)
    calls = []

    def fake_hvg(adata, n_top_genes):
        calls.append((adata.uns["slide_label"], n_top_genes))
        if adata.uns["slide_label"] == "train":
            adata.var["highly_variable"] = [True, False, True]
        else:
            adata.var["highly_variable"] = [False, True, True]

    monkeypatch.setattr(
        "prior_AddOn.build_starmap_plus_mouse_mapping._read_h5ad",
        lambda path: fake_adatas[path],
    )
    monkeypatch.setattr(
        "prior_AddOn.build_starmap_plus_mouse_mapping._compute_highly_variable_genes",
        fake_hvg,
    )

    source_panel = load_starmap_plus_source_panel(tmp_path, n_top_genes=2)

    assert source_panel == ["C"]
    assert calls == [("train", 2), ("test", 2)]


def test_load_starmap_plus_source_panel_rejects_var_name_mismatch(tmp_path, monkeypatch):
    training_path = tmp_path / DEFAULT_TRAINING_SLIDE
    testing_path = tmp_path / DEFAULT_TESTING_SLIDE
    fake_adatas = {
        training_path: _FakeAdata(["A", "B"], highly_variable=[True, True]),
        testing_path: _FakeAdata(["B", "A"], highly_variable=[True, True]),
    }
    _touch_slides(fake_adatas)
    monkeypatch.setattr(
        "prior_AddOn.build_starmap_plus_mouse_mapping._read_h5ad",
        lambda path: fake_adatas[path],
    )

    with pytest.raises(ValueError, match="identical var_names"):
        load_starmap_plus_source_panel(tmp_path, n_top_genes=2)


def test_build_prior_cache_writes_auto_cache_for_later_offline_load(tmp_path):
    root = _write_fake_static_priors(tmp_path / "gene_embeddings")
    _write_mouse_mapping_cache(root)
    source_panel = ["Brca1", "Tp53"]

    first = build_prior_cache(
        source_panel,
        embedding_root=root,
        allow_network=False,
        use_aligned_cache=False,
    )

    auto_cache_files = list((root / "processed" / "aligned").glob("auto_mouse_*_scgpt.pt"))
    assert len(auto_cache_files) == 1
    assert first["scgpt"]["found_mask"].tolist() == [True, True]
    _delete_static_prior_files(root)

    second = load_static_gene_prior(
        source_panel,
        species="mouse",
        root=root,
        models=("scgpt", "geneformer"),
        allow_network=False,
    )

    assert second["scgpt"]["found_mask"].tolist() == [True, True]
    assert torch.equal(second["geneformer"]["embeddings"], first["geneformer"]["embeddings"])


def test_write_mapping_reports_outputs_stable_tables(tmp_path):
    source_panel = ["Brca1", "BadGene"]
    priors = {
        "scgpt": {
            "mapping_table": [
                {
                    "row_index": 0,
                    "model": "scgpt",
                    "input_gene": "Brca1",
                    "normalized_input": "Brca1",
                    "species": "mouse",
                    "mapped_id": "ENSG00000012048",
                    "mapped_symbol": "BRCA1",
                    "token_id": 12,
                    "embedding_index": 2,
                    "status": "mapped",
                    "reason": "mouse_to_human_ortholog",
                },
                {
                    "row_index": 1,
                    "model": "scgpt",
                    "input_gene": "BadGene",
                    "normalized_input": "BadGene",
                    "species": "mouse",
                    "status": "unmapped",
                    "reason": "mouse_mapping_not_in_cache",
                },
            ],
            "coverage": {
                "model": "scgpt",
                "species": "mouse",
                "n_features": 2,
                "n_found": 1,
                "coverage": 0.5,
                "by_status": {"mapped": 1, "unmapped": 1},
            },
        }
    }

    write_mapping_reports(source_panel, priors, tmp_path)

    assert _read_tsv(tmp_path / "source_panel.tsv") == [
        {"row_index": "0", "gene": "Brca1"},
        {"row_index": "1", "gene": "BadGene"},
    ]
    long_rows = _read_tsv(tmp_path / "mapping_table_long.tsv")
    assert long_rows[0]["source_gene"] == "Brca1"
    assert long_rows[0]["mapped_symbol"] == "BRCA1"
    assert long_rows[1]["source_gene"] == "BadGene"
    assert long_rows[1]["status"] == "unmapped"
    assert _read_tsv(tmp_path / "unmapped_or_missing.tsv") == [long_rows[1]]

    with (tmp_path / "coverage_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["source_panel_size"] == 2
    assert summary["models"]["scgpt"]["coverage"] == 0.5


class _FakeAdata:
    def __init__(self, genes, highly_variable=None, slide_label=None):
        self.var_names = pd.Index(genes)
        self.var = pd.DataFrame(index=genes)
        if highly_variable is not None:
            self.var["highly_variable"] = highly_variable
        self.n_vars = len(genes)
        self.uns = {}
        if slide_label is not None:
            self.uns["slide_label"] = slide_label


def _touch_slides(fake_adatas):
    for path in fake_adatas:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _read_tsv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_fake_static_priors(root):
    processed = root / "processed"
    processed.mkdir(parents=True)

    torch.save(
        {
            "model": "scgpt",
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "genes": ["BRCA1", "TP53"],
            "gene_to_idx": {"BRCA1": 0, "TP53": 1},
            "token_ids": {"BRCA1": 12, "TP53": 10},
        },
        processed / "scgpt_static.pt",
    )

    torch.save(
        {
            "model": "geneformer",
            "embeddings": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "genes": ["ENSG00000012048", "ENSG00000141510"],
            "ensembl_to_idx": {"ENSG00000012048": 0, "ENSG00000141510": 1},
            "token_ids": {"ENSG00000012048": 6, "ENSG00000141510": 4},
            "gene_name_to_ensembl": {
                "BRCA1": "ENSG00000012048",
                "TP53": "ENSG00000141510",
            },
            "ensembl_mapping": {
                "BRCA1": "ENSG00000012048",
                "TP53": "ENSG00000141510",
            },
            "ensembl_to_gene_names": {
                "ENSG00000012048": ["BRCA1"],
                "ENSG00000141510": ["TP53"],
            },
        },
        processed / "geneformer_v2_316m_static.pt",
    )
    return root


def _write_mouse_mapping_cache(root):
    path = root / "processed" / "mapping_cache.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_species",
        "input_gene",
        "normalized_gene",
        "human_ensembl",
        "human_symbol",
        "status",
        "reason",
        "updated_at",
    ]
    rows = [
        {
            "source_species": "mouse",
            "input_gene": "Brca1",
            "normalized_gene": "Brca1",
            "human_ensembl": "ENSG00000012048",
            "human_symbol": "BRCA1",
            "status": "mapped",
            "reason": "",
            "updated_at": "2026-05-24T00:00:00+00:00",
        },
        {
            "source_species": "mouse",
            "input_gene": "Tp53",
            "normalized_gene": "Tp53",
            "human_ensembl": "ENSG00000141510",
            "human_symbol": "TP53",
            "status": "mapped",
            "reason": "",
            "updated_at": "2026-05-24T00:00:00+00:00",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _delete_static_prior_files(root):
    (root / "processed" / "scgpt_static.pt").unlink()
    (root / "processed" / "geneformer_v2_316m_static.pt").unlink()
