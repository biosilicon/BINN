from __future__ import absolute_import

import torch
from torch import nn


class GenePriorPooling(nn.Module):
    def __init__(
        self,
        gene_embedding_dim,
        output_dim,
        hidden_dim=128,
        mode="sigmoid",
        expr_transform="identity",
    ):
        super(GenePriorPooling, self).__init__()

        mode = str(mode).lower()
        if mode not in {"sigmoid", "softmax"}:
            raise ValueError("mode must be 'sigmoid' or 'softmax'.")

        expr_transform = str(expr_transform).lower()
        if expr_transform not in {"identity", "log1p"}:
            raise ValueError("expr_transform must be 'identity' or 'log1p'.")

        self.mode = mode
        self.expr_transform = expr_transform
        self.gene_projection = nn.Linear(gene_embedding_dim, output_dim, bias=False)
        self.gene_score = nn.Linear(gene_embedding_dim, hidden_dim, bias=False)
        self.expression_score = nn.Linear(1, hidden_dim, bias=False)
        self.context_score = nn.Linear(output_dim, hidden_dim, bias=True)
        self.score_out = nn.Linear(hidden_dim, 1, bias=True)

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

        gene_hidden = self.gene_score(gene_embeddings).unsqueeze(0)
        expression_hidden = self.expression_score(transformed_expression.unsqueeze(-1))
        context_hidden = self.context_score(h_niche).unsqueeze(1)
        scores = self.score_out(torch.tanh(gene_hidden + expression_hidden + context_hidden)).squeeze(-1)

        if self.mode == "softmax":
            weights = torch.softmax(scores, dim=1)
            weighted_expression = weights
        else:
            weights = torch.sigmoid(scores)
            weighted_expression = weights * transformed_expression

        projected_genes = self.gene_projection(gene_embeddings)
        z_prior = torch.matmul(weighted_expression, projected_genes)
        return z_prior, weights

    def _transform_expression(self, expression):
        if self.expr_transform == "log1p":
            return torch.log1p(expression.clamp_min(0))
        return expression
