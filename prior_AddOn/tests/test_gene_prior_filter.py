from __future__ import annotations

import numpy as np
import pytest
import torch

from prior_AddOn.gene_prior_filter import filter_dataset_by_gene_prior


def test_filter_dataset_by_gene_prior_updates_ad_mouse_like_dataset():
    dataset = _ad_mouse_like_dataset()
    priors = {
        "toy_model": {
            "embeddings": torch.arange(8, dtype=torch.float32).view(4, 2),
            "found_mask": torch.tensor([True, False, True, False]),
            "mapping_table": [
                {"status": "mapped", "gene": "A"},
                {"status": "unmapped", "gene": "B"},
                {"status": "mapped", "gene": "C"},
                {"status": "unmapped", "gene": "D"},
            ],
            "coverage": {
                "model": "toy_model",
                "species": "human",
                "n_features": 4,
                "n_found": 2,
                "coverage": 0.5,
                "by_status": {"mapped": 2, "unmapped": 2},
            },
        }
    }

    dataset, filtered_priors, filter_info = filter_dataset_by_gene_prior(
        dataset,
        priors,
        prior_model="toy_model",
    )

    assert dataset.source_panel.tolist() == ["A", "C"]
    assert dataset.rna_length == 2
    assert dataset.rna_mask.tolist() == [False, True, False, True, False]
    assert dataset.training[0][0].tolist() == [1.0, 3.0]
    assert dataset.training[0][3].shape == (2, 2)
    assert dataset.val[0][0].tolist() == [21.0, 23.0]
    assert filtered_priors["toy_model"]["embeddings"].shape == (2, 2)
    assert filtered_priors["toy_model"]["found_mask"].tolist() == [True, True]
    assert filtered_priors["toy_model"]["coverage"]["coverage"] == 1.0
    assert filter_info["removed_genes"] == ["B", "D"]
    assert [row["gene"] for row in filter_info["removed_mapping_table"]] == ["B", "D"]


def test_filter_dataset_by_gene_prior_updates_sma_like_dataset():
    dataset = _sma_like_dataset()
    priors = {
        "toy_model": {
            "embeddings": torch.randn(3, 2),
            "found_mask": torch.tensor([False, True, True]),
            "mapping_table": [
                {"status": "unmapped", "gene": "A"},
                {"status": "mapped", "gene": "B"},
                {"status": "mapped", "gene": "C"},
            ],
        }
    }

    dataset, filtered_priors, _ = filter_dataset_by_gene_prior(dataset, priors)

    assert dataset.source_panel == ["B", "C"]
    assert dataset.rna_length == 2
    assert dataset.training[0][1].tolist() == [2.0, 3.0]
    assert dataset.training[0][3].shape == (2, 2)
    assert filtered_priors["toy_model"]["embeddings"].shape == (2, 2)


def test_filter_dataset_by_gene_prior_requires_indices_for_unknown_dataset():
    dataset = type("CustomDataset", (), {})()
    dataset.source_panel = ["A"]
    dataset.training = [(np.array([1.0]), np.array([[1.0]]))]
    priors = {"toy_model": {"found_mask": torch.tensor([True])}}

    with pytest.raises(ValueError, match="source_index and neighbor_index"):
        filter_dataset_by_gene_prior(dataset, priors)


def test_nichetrans_registers_teacher_embedding_from_selected_prior():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    embeddings = torch.tensor(
        [
            [3.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 12.0, 0.0],
            [1.0, 2.0, 2.0, 0.0, 0.0],
        ]
    )
    priors = {
        "geneformer": {
            "embeddings": embeddings,
            "found_mask": torch.tensor([True, True, True]),
        }
    }

    model = NicheTrans(source_length=3, target_length=1, priors=priors)

    assert model.prior_model == "geneformer"
    assert hasattr(model, "teacher_embedding")
    assert model.teacher_embedding.shape == (3, 5)
    assert not model.teacher_embedding.requires_grad
    assert torch.allclose(model.teacher_embedding, embeddings)
    assert "teacher_embedding" in model.state_dict()


def test_nichetrans_normalizes_teacher_embedding_to_unit_norm():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(
        source_length=3,
        target_length=1,
        priors=_fixed_norm_priors(),
        normalize_prior_embedding=True,
    )

    assert torch.allclose(
        model.teacher_embedding.norm(p=2, dim=1),
        torch.ones(3),
    )


def test_nichetrans_normalizes_teacher_embedding_to_custom_norm():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(
        source_length=3,
        target_length=1,
        priors=_fixed_norm_priors(),
        normalize_prior_embedding=True,
        prior_embedding_norm=2.5,
    )

    assert torch.allclose(
        model.teacher_embedding.norm(p=2, dim=1),
        torch.full((3,), 2.5),
    )


def test_nichetrans_rejects_zero_norm_teacher_embedding_when_normalizing():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 2.0],
                ]
            ),
            "found_mask": torch.tensor([True, True, True]),
        }
    }

    with pytest.raises(ValueError, match="zero-norm rows"):
        NicheTrans(
            source_length=3,
            target_length=1,
            priors=priors,
            normalize_prior_embedding=True,
        )


def test_nichetrans_requires_positive_prior_embedding_norm():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    with pytest.raises(ValueError, match="positive finite"):
        NicheTrans(
            source_length=3,
            target_length=1,
            priors=_fixed_norm_priors(),
            normalize_prior_embedding=True,
            prior_embedding_norm=0.0,
        )


def test_nichetrans_has_empty_teacher_embedding_without_priors():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(source_length=3, target_length=1)

    assert hasattr(model, "teacher_embedding")
    assert model.teacher_embedding is None
    assert "teacher_embedding" not in model.state_dict()


def test_nichetrans_requires_prior_model_when_multiple_priors_are_provided():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "scgpt": {"embeddings": torch.randn(2, 3), "found_mask": torch.ones(2, dtype=torch.bool)},
        "geneformer": {"embeddings": torch.randn(2, 4), "found_mask": torch.ones(2, dtype=torch.bool)},
    }

    with pytest.raises(ValueError, match="Pass prior_model explicitly"):
        NicheTrans(source_length=2, target_length=1, priors=priors)


def test_nichetrans_rejects_unfiltered_prior():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, False, True]),
        }
    }

    with pytest.raises(ValueError, match="unmatched genes"):
        NicheTrans(source_length=3, target_length=1, priors=priors)


def test_nichetrans_qkv_prior_pooling_forward_returns_attention_and_gate():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(
        source_length=3,
        target_length=2,
        priors=_toy_priors(),
        prior_pooling_mode="qkv",
    )
    model.eval()
    source = torch.rand(2, 3)
    source_neighbor = torch.rand(2, 2, 3)

    with torch.no_grad():
        output, prior_info = model(source, source_neighbor, return_prior=True)

    assert output.shape == (2, 2)
    assert prior_info["prior_weights"].shape == (2, 3)
    assert torch.allclose(
        prior_info["prior_weights"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert prior_info["h_cell0"].shape == (2, 256)
    assert prior_info["h_niche"].shape == (2, 256)
    assert prior_info["z_prior"].shape == (2, 256)
    assert prior_info["prior_gate"].shape == (2, 256)
    assert torch.all(prior_info["prior_gate"] >= 0)
    assert torch.all(prior_info["prior_gate"] <= 1)


def test_qkv_prior_pooling_expression_modulates_value_not_attention():
    from model.gene_prior_pooling import GenePriorPooling

    pooling = GenePriorPooling(
        gene_embedding_dim=5,
        output_dim=256,
        hidden_dim=16,
        mode="qkv",
    )
    pooling.eval()
    gene_embeddings = torch.randn(3, 5)
    h_niche = torch.randn(2, 256)
    expression_a = torch.ones(2, 3)
    expression_b = expression_a.clone()
    expression_b[:, 1] = 4.0

    with torch.no_grad():
        z_prior_a, weights_a = pooling(expression_a, gene_embeddings, h_niche)
        z_prior_b, weights_b = pooling(expression_b, gene_embeddings, h_niche)

    assert torch.allclose(weights_a, weights_b)
    assert not torch.allclose(z_prior_a, z_prior_b)


def test_nichetrans_prior_pooling_requires_priors():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    with pytest.raises(ValueError, match="requires priors"):
        NicheTrans(source_length=3, target_length=1, prior_pooling_mode="qkv")


def test_nichetrans_prior_pooling_none_keeps_default_forward_interface():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(source_length=3, target_length=2, prior_pooling_mode="none")
    model.eval()
    source = torch.rand(2, 3)
    source_neighbor = torch.rand(2, 2, 3)

    with torch.no_grad():
        output = model(source, source_neighbor)

    assert output.shape == (2, 2)
    assert model.prior_pooling is None
    assert model.prior_fusion is None


def _toy_priors():
    return {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, True, True]),
        }
    }


def _fixed_norm_priors():
    return {
        "geneformer": {
            "embeddings": torch.tensor(
                [
                    [3.0, 4.0],
                    [5.0, 12.0],
                    [1.0, 2.0],
                ]
            ),
            "found_mask": torch.tensor([True, True, True]),
        }
    }


def _ad_mouse_like_dataset():
    dataset = type("AD_Mouse", (), {})()
    dataset.source_panel = np.array(["A", "B", "C", "D"])
    dataset.rna_length = 4
    dataset.rna_mask = np.array([False, True, True, True, True])
    dataset.training = [
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            np.array([[1.0], [0.0]]),
            "train",
        )
    ]
    dataset.testing = [
        (
            np.array([11.0, 12.0, 13.0, 14.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([[11.0, 12.0, 13.0, 14.0]]),
            np.array([[0.0]]),
            "test",
        )
    ]
    dataset.val = [
        (
            np.array([21.0, 22.0, 23.0, 24.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([[21.0, 22.0, 23.0, 24.0]]),
            np.array([[0.0]]),
            "val",
        )
    ]
    return dataset


def _sma_like_dataset():
    dataset = type("SMA", (), {})()
    dataset.source_panel = ["A", "B", "C"]
    dataset.rna_length = 3
    dataset.training = [
        (
            "img.png",
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[0.0], [1.0]]),
            "sample",
        )
    ]
    dataset.testing = []
    return dataset
