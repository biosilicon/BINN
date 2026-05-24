from __future__ import absolute_import

import math

import torch
from torch import nn


class GenePriorPooling(nn.Module):
    def __init__(
        self,
        gene_embedding_dim,
        output_dim,
        hidden_dim=128,
        mode="qkv",
        expr_transform="identity",
    ):
        super(GenePriorPooling, self).__init__()

        mode = str(mode).lower()
        if mode != "qkv":
            raise ValueError("mode must be 'qkv'.")

        expr_transform = str(expr_transform).lower()
        if expr_transform not in {"identity", "log1p"}:
            raise ValueError("expr_transform must be 'identity' or 'log1p'.")

        self.mode = mode
        self.expr_transform = expr_transform
        self.scale = 1.0 / math.sqrt(hidden_dim)
        self.query_projection = nn.Linear(output_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(gene_embedding_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(gene_embedding_dim, output_dim, bias=False)

    def forward(self, expression, gene_embeddings, h_niche):
        if expression.ndim != 2:
            raise ValueError(f"expression must have shape [batch, genes], got {tuple(expression.shape)}.")
        if gene_embeddings.ndim != 2:
            raise ValueError(
                f"gene_embeddings must have shape [genes, dim], got {tuple(gene_embeddings.shape)}."
            )
        if h_niche.ndim != 2:
            raise ValueError(f"h_niche must have shape [batch, dim], got {tuple(h_niche.shape)}.")
        if expression.size(1) != gene_embeddings.size(0):
            raise ValueError(
                "expression gene dimension must match gene_embeddings rows: "
                f"{expression.size(1)} != {gene_embeddings.size(0)}."
            )
        if expression.size(0) != h_niche.size(0):
            raise ValueError(
                "expression batch dimension must match h_niche batch dimension: "
                f"{expression.size(0)} != {h_niche.size(0)}."
            )

        expression = expression.to(dtype=gene_embeddings.dtype)
        h_niche = h_niche.to(dtype=gene_embeddings.dtype)
        transformed_expression = self._transform_expression(expression)

        query = self.query_projection(h_niche)
        key = self.key_projection(gene_embeddings)
        value = self.value_projection(gene_embeddings)
        scores = torch.matmul(query, key.transpose(0, 1)) * self.scale
        weights = torch.softmax(scores, dim=1)
        z_prior = torch.matmul(weights * transformed_expression, value)
        return z_prior, weights

    def _transform_expression(self, expression):
        if self.expr_transform == "log1p":
            return torch.log1p(expression.clamp_min(0))
        return expression


class GatedPriorFusion(nn.Module):
    def __init__(self, dim):
        super(GatedPriorFusion, self).__init__()

        self.cell_norm = nn.LayerNorm(dim)
        self.niche_norm = nn.LayerNorm(dim)
        self.prior_norm = nn.LayerNorm(dim)
        self.delta = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Linear(dim * 3, dim)
        self.output_norm = nn.LayerNorm(dim)

        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, h_cell0, h_niche, z_prior):
        fusion_input = torch.cat(
            [
                self.cell_norm(h_cell0),
                self.niche_norm(h_niche),
                self.prior_norm(z_prior),
            ],
            dim=1,
        )
        delta = self.delta(fusion_input)
        gate = torch.sigmoid(self.gate(fusion_input))
        fused = self.output_norm(h_niche + gate * delta)
        return fused, gate
