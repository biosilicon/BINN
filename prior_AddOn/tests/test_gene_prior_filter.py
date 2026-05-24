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

    priors = {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, True, True]),
        }
    }

    model = NicheTrans(source_length=3, target_length=1, priors=priors)

    assert model.prior_model == "geneformer"
    assert model.gene_prior_mode == "niche_conditioned"
    assert hasattr(model, "teacher_embedding")
    assert model.teacher_embedding.shape == (3, 5)
    assert not model.teacher_embedding.requires_grad
    assert model.use_gene_prior is True
    assert isinstance(model.gene_prior_projection, torch.nn.Linear)
    assert model.gene_prior_projection.in_features == 5
    assert model.gene_prior_projection.out_features == model.fea_size
    assert isinstance(model.gene_prior_pooling, torch.nn.Module)
    assert torch.sigmoid(model.gene_prior_gate).max().item() < 0.02
    assert "teacher_embedding" in model.state_dict()


def test_nichetrans_has_empty_teacher_embedding_without_priors():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(source_length=3, target_length=1)

    assert hasattr(model, "teacher_embedding")
    assert model.teacher_embedding is None
    assert model.use_gene_prior is False
    assert model.gene_prior_projection is None
    assert model.gene_prior_gate is None
    assert model.gene_prior_pooling is None
    assert "teacher_embedding" not in model.state_dict()
    assert not any(key.startswith("gene_prior_projection") for key in model.state_dict())
    assert "gene_prior_gate" not in model.state_dict()
    assert not any(key.startswith("gene_prior_pooling") for key in model.state_dict())


def test_nichetrans_forward_runs_without_priors_and_state_dict_stays_baseline():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    model = NicheTrans(source_length=3, target_length=2, noise_rate=0.0, dropout_rate=0.0)
    model.eval()

    source = torch.tensor([[1.0, 0.0, 2.0], [0.5, 1.0, 0.0]])
    source_neighbor = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )

    with torch.no_grad():
        output = model(source, source_neighbor)

    assert output.shape == (2, 2)
    state_keys = model.state_dict().keys()
    assert "teacher_embedding" not in state_keys
    assert not any(key.startswith("gene_prior_") for key in state_keys)


def test_nichetrans_forward_runs_with_gene_prior_projection():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, True, True]),
        }
    }
    model = NicheTrans(
        source_length=3,
        target_length=2,
        noise_rate=0.0,
        dropout_rate=0.0,
        priors=priors,
    )
    model.eval()

    source = torch.tensor([[1.0, 0.0, 2.0], [0.5, 1.0, 0.0]])
    source_neighbor = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )

    with torch.no_grad():
        output = model(source, source_neighbor)

    assert output.shape == (2, 2)


def test_nichetrans_gene_prior_features_use_normalized_expression_weights():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 3.0],
                ]
            ),
            "found_mask": torch.tensor([True, True, True]),
        }
    }
    model = NicheTrans(
        source_length=3,
        target_length=1,
        noise_rate=0.0,
        dropout_rate=0.0,
        priors=priors,
        gene_prior_mode="expression_weighted",
    )
    with torch.no_grad():
        model.gene_prior_projection.weight.zero_()
        model.gene_prior_projection.bias.zero_()
        model.gene_prior_projection.weight[0, 0] = 1.0
        model.gene_prior_projection.weight[1, 1] = 1.0

    omic_data = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 3.0],
        ]
    )

    projected = model._compute_gene_prior_features(omic_data)

    expected = torch.tensor(
        [
            [0.5, 1.0],
            [0.0, 0.0],
            [3.0, 3.0],
        ]
    )
    assert torch.allclose(projected[:, :2], expected)
    assert torch.allclose(projected[:, 2:], torch.zeros_like(projected[:, 2:]))


def test_nichetrans_none_mode_registers_embedding_but_skips_prior_branch():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, True, True]),
        }
    }
    model = NicheTrans(
        source_length=3,
        target_length=2,
        noise_rate=0.0,
        dropout_rate=0.0,
        priors=priors,
        gene_prior_mode="none",
    )
    model.eval()

    assert model.teacher_embedding.shape == (3, 5)
    assert model.use_gene_prior is False
    assert model.gene_prior_projection is None
    assert model.gene_prior_gate is None
    assert model.gene_prior_pooling is None
    assert not any(key.startswith("gene_prior_projection") for key in model.state_dict())
    assert "gene_prior_gate" not in model.state_dict()
    assert not any(key.startswith("gene_prior_pooling") for key in model.state_dict())

    source = torch.tensor([[1.0, 0.0, 2.0], [0.5, 1.0, 0.0]])
    source_neighbor = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [2.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    with torch.no_grad():
        output = model(source, source_neighbor)

    assert output.shape == (2, 2)


@pytest.mark.parametrize("gene_prior_mode", ["learnable_no_niche", "niche_conditioned"])
def test_nichetrans_learnable_prior_modes_are_finite_with_zero_and_negative_expression(gene_prior_mode):
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    priors = {
        "geneformer": {
            "embeddings": torch.randn(3, 5),
            "found_mask": torch.tensor([True, True, True]),
        }
    }
    model = NicheTrans(
        source_length=3,
        target_length=2,
        noise_rate=0.0,
        dropout_rate=0.0,
        priors=priors,
        gene_prior_mode=gene_prior_mode,
    )
    model.eval()

    source = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 3.0]])
    source_neighbor = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [-2.0, 0.0, 0.0]],
        ]
    )

    with torch.no_grad():
        output = model(source, source_neighbor)

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_nichetrans_embedding_controls_random_and_shuffled():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    priors = {
        "geneformer": {
            "embeddings": embeddings,
            "found_mask": torch.tensor([True, True, True, True]),
        }
    }

    torch.manual_seed(1)
    shuffled = NicheTrans(
        source_length=4,
        target_length=1,
        priors=priors,
        gene_prior_mode="expression_weighted",
        gene_prior_embedding_control="shuffled",
    )
    shuffled_rows = sorted(tuple(row.tolist()) for row in shuffled.teacher_embedding)
    original_rows = sorted(tuple(row.tolist()) for row in embeddings)
    assert shuffled_rows == original_rows
    assert not torch.equal(shuffled.teacher_embedding, embeddings)

    torch.manual_seed(1)
    random_control = NicheTrans(
        source_length=4,
        target_length=1,
        priors=priors,
        gene_prior_mode="expression_weighted",
        gene_prior_embedding_control="random",
    )
    assert random_control.teacher_embedding.shape == embeddings.shape
    assert not torch.equal(random_control.teacher_embedding, embeddings)
    assert torch.allclose(random_control.teacher_embedding.mean(), embeddings.mean(), atol=1e-6)
    assert torch.allclose(
        random_control.teacher_embedding.std(unbiased=False),
        embeddings.std(unbiased=False),
        atol=1e-6,
    )


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
