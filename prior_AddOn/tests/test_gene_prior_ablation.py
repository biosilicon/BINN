from __future__ import annotations

import pytest
import torch

from prior_AddOn.gene_prior_ablation import randomize_static_gene_prior


def test_randomize_static_gene_prior_is_deterministic_by_seed():
    priors = _toy_priors()

    first = randomize_static_gene_prior(priors, prior_model="scgpt", seed=7)
    second = randomize_static_gene_prior(priors, prior_model="scgpt", seed=7)
    changed = randomize_static_gene_prior(priors, prior_model="scgpt", seed=8)

    assert torch.equal(first["scgpt"]["embeddings"], second["scgpt"]["embeddings"])
    assert not torch.equal(first["scgpt"]["embeddings"], changed["scgpt"]["embeddings"])


def test_randomize_static_gene_prior_does_not_mutate_original_prior():
    priors = _toy_priors()
    original_embeddings = priors["scgpt"]["embeddings"].clone()

    randomized = randomize_static_gene_prior(priors, prior_model="scgpt", seed=11)

    assert randomized is not priors
    assert randomized["scgpt"] is not priors["scgpt"]
    assert torch.equal(priors["scgpt"]["embeddings"], original_embeddings)
    assert "ablation" not in priors["scgpt"]
    assert not torch.equal(randomized["scgpt"]["embeddings"], original_embeddings)


def test_randomize_static_gene_prior_preserves_metadata_shape_and_dtype():
    priors = _toy_priors()

    randomized = randomize_static_gene_prior(priors, prior_model="scgpt", seed=13)

    assert randomized["scgpt"]["embeddings"].shape == priors["scgpt"]["embeddings"].shape
    assert randomized["scgpt"]["embeddings"].dtype == priors["scgpt"]["embeddings"].dtype
    assert torch.equal(randomized["scgpt"]["found_mask"], priors["scgpt"]["found_mask"])
    assert randomized["scgpt"]["mapping_table"] == priors["scgpt"]["mapping_table"]
    assert randomized["scgpt"]["coverage"] == priors["scgpt"]["coverage"]
    assert randomized["scgpt"]["ablation"] == {
        "type": "random_embedding",
        "seed": 13,
        "distribution": "match_mean_std",
        "source_model": "scgpt",
        "embedding_shape": [3, 4],
    }


def test_randomize_static_gene_prior_does_not_advance_global_torch_rng():
    priors = _toy_priors()
    torch.manual_seed(123)
    before = torch.get_rng_state()

    randomize_static_gene_prior(priors, prior_model="scgpt", seed=17)

    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_randomize_static_gene_prior_rejects_unknown_model():
    with pytest.raises(ValueError, match="was not found"):
        randomize_static_gene_prior(_toy_priors(), prior_model="missing", seed=1)


def test_nichetrans_accepts_randomized_prior_with_qkv_pooling():
    pytest.importorskip("einops")
    from model.nicheTrans import NicheTrans

    randomized = randomize_static_gene_prior(_toy_priors(), prior_model="scgpt", seed=19)
    model = NicheTrans(
        source_length=3,
        target_length=2,
        priors=randomized,
        prior_model="scgpt",
        prior_pooling_mode="qkv",
    )
    model.eval()

    with torch.no_grad():
        output, prior_info = model(torch.rand(2, 3), torch.rand(2, 2, 3), return_prior=True)

    assert output.shape == (2, 2)
    assert prior_info["prior_weights"].shape == (2, 3)
    assert prior_info["prior_gate"].shape == (2, 256)


def _toy_priors():
    return {
        "scgpt": {
            "embeddings": torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            "found_mask": torch.tensor([True, True, True]),
            "mapping_table": [
                {"status": "mapped", "gene": "A"},
                {"status": "mapped", "gene": "B"},
                {"status": "mapped", "gene": "C"},
            ],
            "coverage": {
                "model": "scgpt",
                "species": "human",
                "n_features": 3,
                "n_found": 3,
                "coverage": 1.0,
                "by_status": {"mapped": 3},
            },
        }
    }
