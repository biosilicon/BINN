from __future__ import annotations

import csv

import torch

from prior_AddOn.gene_embedding_loader import load_static_gene_prior


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


def _write_mouse_mapping_cache(root):
    path = root / "processed" / "mapping_cache.tsv"
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

