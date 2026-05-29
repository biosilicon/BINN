from __future__ import annotations

from typing import Any

import torch


def randomize_static_gene_prior(
    priors: dict[str, dict[str, Any]],
    prior_model: str,
    seed: int,
    distribution: str = "match_mean_std",
) -> dict[str, dict[str, Any]]:
    """Return a copy of priors with one model's embeddings replaced by noise.

    The random numbers are generated with a private CPU generator so ablation
    seeds do not advance the global training RNG state.
    """

    if prior_model not in priors:
        raise ValueError(
            f"prior_model {prior_model!r} was not found in priors. "
            f"Available models: {sorted(priors)}"
        )

    distribution = str(distribution).lower()
    if distribution != "match_mean_std":
        raise ValueError("distribution must be 'match_mean_std'.")

    selected_prior = priors[prior_model]
    if "embeddings" not in selected_prior:
        raise ValueError(f"priors[{prior_model!r}] does not contain 'embeddings'.")

    embeddings = torch.as_tensor(selected_prior["embeddings"])
    if embeddings.ndim != 2:
        raise ValueError(
            f"priors[{prior_model!r}]['embeddings'] must be two-dimensional, "
            f"got shape {tuple(embeddings.shape)}."
        )
    if not torch.is_floating_point(embeddings):
        raise ValueError(f"priors[{prior_model!r}]['embeddings'] must be a floating point tensor.")

    randomized = _match_mean_std_random_embeddings(embeddings, seed)

    randomized_priors = {model_name: dict(model_prior) for model_name, model_prior in priors.items()}
    randomized_priors[prior_model] = {
        **randomized_priors[prior_model],
        "embeddings": randomized,
        "ablation": {
            "type": "random_embedding",
            "seed": int(seed),
            "distribution": distribution,
            "source_model": prior_model,
            "embedding_shape": [int(dim) for dim in embeddings.shape],
        },
    }
    return randomized_priors


def _match_mean_std_random_embeddings(embeddings: torch.Tensor, seed: int) -> torch.Tensor:
    stats_tensor = embeddings.detach().to(device="cpu", dtype=torch.float32)
    mean = stats_tensor.mean()
    std = stats_tensor.std(unbiased=False)
    if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(std)):
        raise ValueError("Cannot randomize prior embeddings with non-finite mean or standard deviation.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    randomized = torch.randn(
        tuple(int(dim) for dim in embeddings.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    randomized = randomized * std + mean
    return randomized.to(device=embeddings.device, dtype=embeddings.dtype)
