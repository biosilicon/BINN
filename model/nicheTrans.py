from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *


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
    ):
        super(NicheTrans, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate
        self.prior_model = self._resolve_prior_model(priors, prior_model)

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
        self.use_gene_prior = self.teacher_embedding is not None
        if self.use_gene_prior:
            self.gene_prior_projection = nn.Linear(self.teacher_embedding.size(1), self.fea_size)
        else:
            self.gene_prior_projection = None

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

    def _compute_gene_prior_features(self, omic_data):
        if not self.use_gene_prior:
            raise RuntimeError("Gene prior features were requested, but no teacher embedding is registered.")

        expression = omic_data.clamp(min=0)
        expression_sum = expression.sum(dim=1, keepdim=True)
        weights = expression / expression_sum.clamp_min(1e-12)
        prior_features = weights @ self.teacher_embedding
        return self.gene_prior_projection(prior_features)

    def forward(self, source, source_neighbor):
        b = source.size(0)
        l = source_neighbor.size(1)
        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, l//2, 1), self.token_neigh_2.repeat(1, l//2, 1)], dim=1)

        source = source[:, None, :]
        omic_data = torch.cat([source, source_neighbor], dim=1).view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder(omic_data)
        if self.use_gene_prior:
            gene_prior_features = self._compute_gene_prior_features(omic_data)
            f_omic = f_omic + gene_prior_features
        f_omic = f_omic.view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens

        f_omic = self.non_linear(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f = self.dropout(f_omic[:, 0, :])

        # final prediction
        out = []
        for i in range(self.target_length):
            out.append(self.predict_layers[i](f))
        out = torch.cat(out, dim=1)

        return out
    
