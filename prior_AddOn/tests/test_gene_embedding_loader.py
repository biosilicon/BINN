from __future__ import annotations

import csv

import torch

from prior_AddOn.gene_embedding_loader import GeneResolver, load_static_gene_prior


def test_human_static_prior_alignment(tmp_path):
    root = _write_fake_static_priors(tmp_path)

    source_panel = ["TP53", "ACTB", "ENSG00000141510.17", "chr1:100-200", "UNKNOWN"]
    result = load_static_gene_prior(
        source_panel,
        "human",
        root=root,
        allow_network=False,
        write_aligned=True,
        dataset_key="toy_human",
    )

    assert set(result) == {"scgpt", "geneformer"}
    assert result["scgpt"]["embeddings"].shape == (5, 2)
    assert result["geneformer"]["embeddings"].shape == (5, 3)
    assert result["scgpt"]["found_mask"].tolist() == [True, True, True, False, False]
    assert result["geneformer"]["found_mask"].tolist() == [True, True, True, False, False]
    assert result["scgpt"]["mapping_table"][3]["status"] == "non_gene_feature"
    assert result["geneformer"]["mapping_table"][4]["status"] == "unmapped"
    assert (root / "processed" / "aligned" / "toy_human_scgpt.pt").exists()
    assert (root / "processed" / "aligned" / "toy_human_geneformer.pt").exists()


def test_mouse_cache_to_human_ortholog_alignment(tmp_path):
    root = _write_fake_static_priors(tmp_path)
    _write_mouse_mapping_cache(root)

    source_panel = ["Brca1", "ENSMUSG00000017146"]
    result = load_static_gene_prior(source_panel, "mouse", root=root, allow_network=False)

    assert result["scgpt"]["found_mask"].tolist() == [True, True]
    assert result["geneformer"]["found_mask"].tolist() == [True, True]
    assert result["scgpt"]["mapping_table"][0]["mapped_symbol"] == "BRCA1"
    assert "mouse_to_human_ortholog" in result["scgpt"]["mapping_table"][0]["reason"]
    assert result["geneformer"]["mapping_table"][1]["mapped_id"] == "ENSG00000012048"


def test_dataset_key_cache_loads_without_static_model_files(tmp_path):
    root = _write_fake_static_priors(tmp_path)
    source_panel = ["TP53", "ACTB"]
    first = load_static_gene_prior(
        source_panel,
        "human",
        root=root,
        allow_network=False,
        dataset_key="toy_cache",
    )
    _delete_static_prior_files(root)

    second = load_static_gene_prior(
        source_panel,
        "human",
        root=root,
        allow_network=False,
        dataset_key="toy_cache",
    )

    assert second["scgpt"]["found_mask"].tolist() == first["scgpt"]["found_mask"].tolist()
    assert torch.equal(second["geneformer"]["embeddings"], first["geneformer"]["embeddings"])


def test_auto_cache_loads_without_static_model_files(tmp_path):
    root = _write_fake_static_priors(tmp_path)
    source_panel = ["TP53", "ENSG00000141510.17"]
    load_static_gene_prior(source_panel, "human", root=root, allow_network=False)

    auto_cache_files = list((root / "processed" / "aligned").glob("auto_human_*_scgpt.pt"))
    assert len(auto_cache_files) == 1
    _delete_static_prior_files(root)

    result = load_static_gene_prior(source_panel, "human", root=root, allow_network=False)

    assert result["scgpt"]["found_mask"].tolist() == [True, True]
    assert result["geneformer"]["embeddings"].shape == (2, 3)


def test_dataset_key_cache_miss_when_source_panel_changes(tmp_path):
    root = _write_fake_static_priors(tmp_path)
    load_static_gene_prior(["TP53"], "human", root=root, allow_network=False, dataset_key="panel")

    changed = load_static_gene_prior(
        ["TP53", "ACTB"],
        "human",
        root=root,
        allow_network=False,
        dataset_key="panel",
    )

    assert changed["scgpt"]["embeddings"].shape == (2, 2)
    assert changed["geneformer"]["found_mask"].tolist() == [True, True]


def test_mouse_panel_precache_writes_completed_gene_before_interruption(tmp_path, monkeypatch):
    resolver = GeneResolver(tmp_path, allow_network=True)
    geneformer_dicts = _fake_geneformer_dicts()

    def lookup_symbol(species, symbol):
        if symbol == "StopNow":
            raise KeyboardInterrupt
        return "ENSMUSG00000017146"

    monkeypatch.setattr(resolver, "_lookup_symbol", lookup_symbol)
    monkeypatch.setattr(resolver, "_fetch_homologies", lambda species, gene_id: [_one_to_one("ENSG00000012048")])

    try:
        resolver.resolve_mouse_panel(["Brca1", "StopNow"], geneformer_dicts)
    except KeyboardInterrupt:
        pass

    rows = _read_mapping_cache(tmp_path)
    assert rows["Brca1"]["status"] == "mapped"
    assert rows["Brca1"]["human_ensembl"] == "ENSG00000012048"
    assert "StopNow" not in rows


def test_mouse_panel_precache_resumes_from_existing_cache(tmp_path, monkeypatch):
    _write_mouse_mapping_cache(tmp_path)
    resolver = GeneResolver(tmp_path, allow_network=True)
    geneformer_dicts = _fake_geneformer_dicts()
    looked_up = []

    def lookup_symbol(species, symbol):
        looked_up.append(symbol)
        if symbol == "Brca1":
            raise AssertionError("cached genes should not be looked up again")
        return "ENSMUSG00000000001"

    monkeypatch.setattr(resolver, "_lookup_symbol", lookup_symbol)
    monkeypatch.setattr(resolver, "_fetch_homologies", lambda species, gene_id: [_one_to_one("ENSG00000141510")])

    resolver.resolve_mouse_panel(["Brca1", "Tp53"], geneformer_dicts)

    rows = _read_mapping_cache(tmp_path)
    assert looked_up == ["Tp53"]
    assert rows["Brca1"]["human_ensembl"] == "ENSG00000012048"
    assert rows["Tp53"]["human_ensembl"] == "ENSG00000141510"


def test_failed_mapping_cache_is_not_retried_by_default(tmp_path, monkeypatch):
    _write_mouse_mapping_cache(
        tmp_path,
        rows=[
            {
                "source_species": "mouse",
                "input_gene": "BadGene",
                "normalized_gene": "BadGene",
                "human_ensembl": "",
                "human_symbol": "",
                "status": "unmapped",
                "reason": "ensembl_request_failed:Timeout",
                "updated_at": "2026-05-24T00:00:00+00:00",
            }
        ],
    )
    resolver = GeneResolver(tmp_path, allow_network=True, retry_failed_mappings=False)
    monkeypatch.setattr(resolver, "_lookup_symbol", lambda species, symbol: (_ for _ in ()).throw(AssertionError()))

    resolution = resolver.resolve_mouse_gene("BadGene", _fake_geneformer_dicts())

    assert resolution.status == "unmapped"
    assert resolution.reason == "ensembl_request_failed:Timeout"


def test_failed_mapping_cache_can_be_retried(tmp_path, monkeypatch):
    _write_mouse_mapping_cache(
        tmp_path,
        rows=[
            {
                "source_species": "mouse",
                "input_gene": "BadGene",
                "normalized_gene": "BadGene",
                "human_ensembl": "",
                "human_symbol": "",
                "status": "unmapped",
                "reason": "ensembl_request_failed:Timeout",
                "updated_at": "2026-05-24T00:00:00+00:00",
            }
        ],
    )
    resolver = GeneResolver(tmp_path, allow_network=True, retry_failed_mappings=True)
    monkeypatch.setattr(resolver, "_lookup_symbol", lambda species, symbol: "ENSMUSG00000017146")
    monkeypatch.setattr(resolver, "_fetch_homologies", lambda species, gene_id: [_one_to_one("ENSG00000012048")])

    resolution = resolver.resolve_mouse_gene("BadGene", _fake_geneformer_dicts())

    assert resolution.status == "mapped"
    rows = _read_mapping_cache(tmp_path)
    assert rows["BadGene"]["human_ensembl"] == "ENSG00000012048"


def _write_fake_static_priors(root):
    processed = root / "processed"
    processed.mkdir(parents=True)

    torch.save(
        {
            "model": "scgpt",
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "genes": ["TP53", "ACTB", "BRCA1"],
            "gene_to_idx": {"TP53": 0, "ACTB": 1, "BRCA1": 2},
            "token_ids": {"TP53": 10, "ACTB": 11, "BRCA1": 12},
        },
        processed / "scgpt_static.pt",
    )

    gene_name_to_ensembl = {
        "TP53": "ENSG00000141510",
        "ACTB": "ENSG00000075624",
        "BRCA1": "ENSG00000012048",
    }
    ensembl_to_gene_names = {
        "ENSG00000141510": ["TP53"],
        "ENSG00000075624": ["ACTB"],
        "ENSG00000012048": ["BRCA1"],
    }
    torch.save(
        {
            "model": "geneformer",
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "genes": ["ENSG00000141510", "ENSG00000075624", "ENSG00000012048"],
            "ensembl_to_idx": {
                "ENSG00000141510": 0,
                "ENSG00000075624": 1,
                "ENSG00000012048": 2,
            },
            "token_ids": {
                "ENSG00000141510": 4,
                "ENSG00000075624": 5,
                "ENSG00000012048": 6,
            },
            "gene_name_to_ensembl": gene_name_to_ensembl,
            "ensembl_mapping": {key: value for key, value in gene_name_to_ensembl.items()},
            "ensembl_to_gene_names": ensembl_to_gene_names,
        },
        processed / "geneformer_v2_316m_static.pt",
    )
    return root


def _fake_geneformer_dicts():
    return {
        "ensembl_to_gene_names": {
            "ENSG00000012048": ["BRCA1"],
            "ENSG00000141510": ["TP53"],
        }
    }


def _one_to_one(human_ensembl):
    return {
        "type": "ortholog_one2one",
        "target": {
            "species": "homo_sapiens",
            "id": human_ensembl,
        },
    }


def _delete_static_prior_files(root):
    (root / "processed" / "scgpt_static.pt").unlink()
    (root / "processed" / "geneformer_v2_316m_static.pt").unlink()


def _read_mapping_cache(root):
    path = root / "processed" / "mapping_cache.tsv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["normalized_gene"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def _write_mouse_mapping_cache(root, rows=None):
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
    rows = rows or [
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
            "input_gene": "ENSMUSG00000017146",
            "normalized_gene": "ENSMUSG00000017146",
            "human_ensembl": "ENSG00000012048",
            "human_symbol": "BRCA1",
            "status": "mapped",
            "reason": "",
            "updated_at": "2026-05-24T00:00:00+00:00",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
