from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *


GENE_PRIOR_MODES = {
    "none",
    "expression_weighted",
    "learnable_no_niche",
    "niche_conditioned",
}
GENE_PRIOR_EMBEDDING_CONTROLS = {"pretrained", "random", "shuffled"}


class NetBlock(nn.Module):
    def __init__(self, nlayer: int, dim_list: list, dropout_rate: float, noise_rate: float):

        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(nn.LeakyReLU())
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))
        
    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x


class GenePriorPooling(nn.Module):
    def __init__(self, embedding_dim: int, feature_dim: int, use_niche: bool):
        super(GenePriorPooling, self).__init__()
        self.use_niche = use_niche
        input_dim = embedding_dim + 1 + feature_dim
        if use_niche:
            input_dim += feature_dim

        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, gene_embedding, expression, cell_state, niche_state=None):
        if self.use_niche and niche_state is None:
            raise ValueError("niche_state is required for niche-conditioned gene prior pooling.")

        expression = expression.clamp(min=0)
        expression_sum = expression.sum(dim=1, keepdim=True)
        expression_weights = expression / expression_sum.clamp_min(1e-12)

        batch_size, n_genes = expression.shape
        gene_context = gene_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        cell_context = cell_state.unsqueeze(1).expand(-1, n_genes, -1)
        features = [gene_context, expression_weights.unsqueeze(-1), cell_context]
        if self.use_niche:
            niche_context = niche_state.unsqueeze(1).expand(-1, n_genes, -1)
            features.append(niche_context)

        logits = self.weight_net(torch.cat(features, dim=-1)).squeeze(-1)
        logits = logits.masked_fill(expression <= 0, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * (expression_sum > 0).to(weights.dtype)

        return weights @ gene_embedding, weights


# NicheTrans with spatial information only
class NicheTrans(nn.Module):
    def __init__(
        self,
        source_length=877,
        target_length=137,
        noise_rate=0.2,
        dropout_rate=0.1,
        priors=None,
        prior_model=None,
        gene_prior_mode=None,
        gene_prior_embedding_control="pretrained",
        gene_prior_gate_init=-4.0,
    ):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.prior_model = self._resolve_prior_model(priors, prior_model)
        self.gene_prior_mode = self._resolve_gene_prior_mode(priors, gene_prior_mode)
        self.gene_prior_embedding_control = self._resolve_gene_prior_embedding_control(
            gene_prior_embedding_control
        )
        if self.gene_prior_mode != "none" and priors is None:
            raise ValueError("gene_prior_mode requires priors unless it is set to 'none'.")

        self.fea_size, self.img_size = 256, 128

        ###############
        # omics encoder
        self.encoder = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)
    
        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)

        ##############
        # prediction layers
        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(nn.Linear(self.fea_size, 128),
                             nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                             nn.Linear(128, 1, bias=True))
            )
        self.predict_layers = nn.ModuleList(predict_net)

        ################
        # others
        self.non_linear = nn.Sequential(nn.Linear(256, 256),
                                        nn.LayerNorm(256),
                                        nn.LeakyReLU())
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout_5 = nn.Dropout(0.5)

        ################
        # initialize tokens for semantic embedding
        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))

        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)

        ################# 
        # teacher embedding from pretrained models
        self.register_buffer("teacher_embedding", None, persistent=False)
        self.teacher_embedding = None
        self._register_teacher_embedding(priors)
        self.use_gene_prior = self.teacher_embedding is not None and self.gene_prior_mode != "none"
        if self.use_gene_prior:
            self.gene_prior_projection = nn.Linear(self.teacher_embedding.size(1), self.fea_size)
            nn.init.zeros_(self.gene_prior_projection.bias)
            self.gene_prior_gate = nn.Parameter(torch.full((self.fea_size,), float(gene_prior_gate_init)))
            if self.gene_prior_mode in {"learnable_no_niche", "niche_conditioned"}:
                self.gene_prior_pooling = GenePriorPooling(
                    embedding_dim=self.teacher_embedding.size(1),
                    feature_dim=self.fea_size,
                    use_niche=self.gene_prior_mode == "niche_conditioned",
                )
            else:
                self.gene_prior_pooling = None
        else:
            self.gene_prior_projection = None
            self.gene_prior_gate = None
            self.gene_prior_pooling = None

    def _resolve_prior_model(self, priors, prior_model):
        if priors is None:
            return prior_model
        if not priors:
            raise ValueError("'priors' must contain at least one prior model.")
        if prior_model is not None:
            if prior_model not in priors:
                raise ValueError(
                    f"prior_model {prior_model!r} was not found in priors. "
                    f"Available models: {sorted(priors)}"
                )
            return prior_model
        if len(priors) == 1:
            return next(iter(priors))
        raise ValueError(
            "Multiple prior models were provided. Pass prior_model explicitly. "
            f"Available models: {sorted(priors)}"
        )

    def _resolve_gene_prior_mode(self, priors, gene_prior_mode):
        if gene_prior_mode is None:
            return "niche_conditioned" if priors is not None else "none"

        mode = str(gene_prior_mode).strip().lower().replace("-", "_")
        aliases = {
            "off": "none",
            "false": "none",
            "disabled": "none",
            "expression": "expression_weighted",
            "expr_weighted": "expression_weighted",
            "learnable": "learnable_no_niche",
            "learnable_without_niche": "learnable_no_niche",
            "niche": "niche_conditioned",
            "niche_condition": "niche_conditioned",
        }
        mode = aliases.get(mode, mode)
        if mode not in GENE_PRIOR_MODES:
            raise ValueError(
                f"Unsupported gene_prior_mode {gene_prior_mode!r}. "
                f"Available modes: {sorted(GENE_PRIOR_MODES)}"
            )
        return mode

    def _resolve_gene_prior_embedding_control(self, embedding_control):
        control = str(embedding_control).strip().lower().replace("-", "_")
        if control not in GENE_PRIOR_EMBEDDING_CONTROLS:
            raise ValueError(
                f"Unsupported gene_prior_embedding_control {embedding_control!r}. "
                f"Available controls: {sorted(GENE_PRIOR_EMBEDDING_CONTROLS)}"
            )
        return control

    def _register_teacher_embedding(self, priors):
        if priors is None:
            self.teacher_embedding = None
            return

        prior = priors[self.prior_model]
        if "embeddings" not in prior:
            raise ValueError(f"priors[{self.prior_model!r}] does not contain 'embeddings'.")

        teacher_embedding = torch.as_tensor(prior["embeddings"], dtype=torch.float32).detach().clone()
        if teacher_embedding.ndim != 2:
            raise ValueError(
                f"priors[{self.prior_model!r}]['embeddings'] must be two-dimensional, "
                f"got shape {tuple(teacher_embedding.shape)}."
            )
        if teacher_embedding.size(0) != self.source_length:
            raise ValueError(
                f"Teacher embedding rows ({teacher_embedding.size(0)}) must match "
                f"source_length ({self.source_length}). Filter the dataset and priors "
                "with filter_dataset_by_gene_prior before creating NicheTrans."
            )
        found_mask = prior.get("found_mask")
        if found_mask is not None:
            found_mask = torch.as_tensor(found_mask, dtype=torch.bool)
            if found_mask.ndim != 1 or found_mask.numel() != self.source_length:
                raise ValueError(
                    f"priors[{self.prior_model!r}]['found_mask'] must have length "
                    f"{self.source_length}."
                )
            if not bool(found_mask.all()):
                raise ValueError(
                    f"priors[{self.prior_model!r}] still contains unmatched genes. "
                    "Run filter_dataset_by_gene_prior before creating NicheTrans."
                )
        teacher_embedding = self._apply_gene_embedding_control(teacher_embedding)
        self.register_buffer("teacher_embedding", None, persistent=True)
        self.teacher_embedding = teacher_embedding

    def _apply_gene_embedding_control(self, teacher_embedding):
        if self.gene_prior_embedding_control == "pretrained":
            return teacher_embedding
        if self.gene_prior_embedding_control == "shuffled":
            if teacher_embedding.size(0) <= 1:
                return teacher_embedding
            permutation = torch.randperm(teacher_embedding.size(0))
            if torch.equal(permutation, torch.arange(teacher_embedding.size(0))):
                permutation = torch.roll(permutation, shifts=1)
            return teacher_embedding[permutation].clone()
        if self.gene_prior_embedding_control == "random":
            mean = teacher_embedding.mean()
            std = teacher_embedding.std(unbiased=False)
            random_embedding = torch.randn_like(teacher_embedding)
            random_embedding = random_embedding - random_embedding.mean()
            random_std = random_embedding.std(unbiased=False)
            if float(random_std) > 0:
                random_embedding = random_embedding / random_std
            return random_embedding * std + mean
        raise RuntimeError("Unexpected gene prior embedding control.")

    def _compute_gene_prior_features(self, omic_data):
        if not self.use_gene_prior:
            raise RuntimeError("Gene prior features were requested, but no teacher embedding is registered.")

        expression = omic_data.clamp(min=0)
        expression_sum = expression.sum(dim=1, keepdim=True)
        weights = expression / expression_sum.clamp_min(1e-12)
        prior_features = weights @ self.teacher_embedding
        return self.gene_prior_projection(prior_features)

    def _build_spatial_tokens(self, n_neighbors):
        return torch.cat(
            [
                self.token_center,
                self.token_neigh_1.repeat(1, n_neighbors // 2, 1),
                self.token_neigh_2.repeat(1, n_neighbors // 2, 1),
            ],
            dim=1,
        )

    def _build_omic_data(self, source, source_neighbor):
        source = source[:, None, :]
        return torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

    def _encode_expression(self, omic_data, batch_size):
        return self.encoder(omic_data).view(batch_size, -1, self.fea_size)

    def _run_niche_encoder(self, encoded_tokens, spatial_tokens):
        f_omic = encoded_tokens + spatial_tokens
        f_omic = self.non_linear(f_omic)
        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic
        return f_omic

    def _compute_refined_cell_state(self, expression, cell_state, niche_state=None):
        if self.gene_prior_mode == "expression_weighted":
            prior_state = self._compute_gene_prior_features(expression)
        elif self.gene_prior_mode in {"learnable_no_niche", "niche_conditioned"}:
            prior_features, _ = self.gene_prior_pooling(
                self.teacher_embedding,
                expression,
                cell_state,
                niche_state,
            )
            prior_state = self.gene_prior_projection(prior_features)
        else:
            raise RuntimeError(f"Unexpected active gene_prior_mode: {self.gene_prior_mode}")

        gate = torch.sigmoid(self.gene_prior_gate).view(1, -1)
        return cell_state + gate * prior_state

    def _run_gene_prior_refinement(self, expression, encoded_tokens, spatial_tokens):
        cell_state = encoded_tokens[:, 0, :]
        niche_state = None
        if self.gene_prior_mode == "niche_conditioned":
            initial_niche_tokens = self._run_niche_encoder(encoded_tokens, spatial_tokens)
            niche_state = initial_niche_tokens[:, 0, :]

        refined_cell_state = self._compute_refined_cell_state(
            expression,
            cell_state,
            niche_state,
        )
        refined_tokens = torch.cat(
            [refined_cell_state[:, None, :], encoded_tokens[:, 1:, :]],
            dim=1,
        )
        return self._run_niche_encoder(refined_tokens, spatial_tokens)

    def _predict(self, features):
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](features))
        return torch.cat(out, dim=1)

    def forward(self, source, source_neighbor):
        b = source.size(0)
        l = source_neighbor.size(1)
        spatial_tokens = self._build_spatial_tokens(l)

        omic_data = self._build_omic_data(source, source_neighbor)
        encoded_tokens = self._encode_expression(omic_data, b)
        if self.use_gene_prior:
            f_omic = self._run_gene_prior_refinement(source, encoded_tokens, spatial_tokens)
        else:
            f_omic = self._run_niche_encoder(encoded_tokens, spatial_tokens)

        f = self.dropout(f_omic[:, 0, :])

        return self._predict(f)
    
