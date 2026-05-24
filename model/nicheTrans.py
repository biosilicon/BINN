from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.gene_prior_pooling import GenePriorPooling, GatedPriorFusion


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
        prior_pooling_mode="none",
        prior_hidden_dim=128,
        prior_expr_transform="identity",
    ):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.prior_model = self._resolve_prior_model(priors, prior_model)
        self.prior_pooling_mode = self._normalize_prior_pooling_mode(prior_pooling_mode)
        self.prior_hidden_dim = prior_hidden_dim
        self.prior_expr_transform = prior_expr_transform

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

        self.register_buffer("teacher_embedding", None, persistent=False)
        self.teacher_embedding = None
        self._register_teacher_embedding(priors)

        self.prior_pooling = None
        self.prior_fusion = None
        self._build_prior_pooling()

    def _normalize_prior_pooling_mode(self, prior_pooling_mode):
        mode = str(prior_pooling_mode).lower()
        if mode not in {"none", "qkv"}:
            raise ValueError("prior_pooling_mode must be 'none' or 'qkv'.")
        return mode

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
        self.register_buffer("teacher_embedding", None, persistent=True)
        self.teacher_embedding = teacher_embedding

    def _build_prior_pooling(self):
        if self.prior_pooling_mode == "none":
            return
        if self.teacher_embedding is None:
            raise ValueError(
                "prior_pooling_mode requires priors. Pass filtered priors or set "
                "prior_pooling_mode='none'."
            )

        self.prior_pooling = GenePriorPooling(
            gene_embedding_dim=self.teacher_embedding.size(1),
            output_dim=self.fea_size,
            hidden_dim=self.prior_hidden_dim,
            mode=self.prior_pooling_mode,
            expr_transform=self.prior_expr_transform,
        )
        self.prior_fusion = GatedPriorFusion(self.fea_size)

    def forward(self, source, source_neighbor, return_prior=False):
        b = source.size(0)
        l = source_neighbor.size(1)
        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, l//2, 1), self.token_neigh_2.repeat(1, l//2, 1)], dim=1)

        source_expression = source
        source_token = source[:, None, :]
        omic_data = torch.cat([source_token, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic_raw = self.encoder(omic_data).view(b, -1, self.fea_size)
        h_cell0 = f_omic_raw[:, 0, :]
        f_omic = f_omic_raw
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        h_niche = f_omic[:, 0, :]
        prior_info = {
            "h_cell0": h_cell0,
            "h_niche": h_niche,
        }

        if self.prior_pooling is not None:
            z_prior, prior_weights = self.prior_pooling(
                source_expression,
                self.teacher_embedding,
                h_niche,
            )
            f, prior_gate = self.prior_fusion(h_cell0, h_niche, z_prior)
            prior_info["z_prior"] = z_prior
            prior_info["prior_weights"] = prior_weights
            prior_info["prior_gate"] = prior_gate
        else:
            f = h_niche

        f = self.dropout(f)

        # final prediction
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        if return_prior:
            return out, prior_info
        return out
    
