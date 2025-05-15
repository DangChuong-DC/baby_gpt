from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0) -> None:
        super(SelfAttention, self).__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        qry: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_bias = None
        # TODO: generate `attn_bias` using `attn_mask`
        if attn_mask is not None:
            B = qry.size(0)
            S_q, S_k = qry.size(-2), key.size(-2)
            attn_bias = torch.zeros(B, S_q, S_k, dtype=qry.dtype, device=qry.device)
            attn_bias = attn_bias.masked_fill_(attn_mask.logical_not(), -math.inf) # (B, S_q, S_k)
            if qry.ndim == 4: # (B, H, S, D)
                attn_bias = attn_bias.unsqueeze_(1)

        scaled_dot_prod = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(key.size(-1)) # (B, H, S_q, S_k)
        if attn_bias is not None:
            scaled_dot_prod += attn_bias
        attn_weights = F.softmax(scaled_dot_prod, dim=-1)
        if self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        output = torch.matmul(attn_weights, val) # (B, H, S_k, D)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        attn_dropout_p: float = 0.0,
        out_dropout_p: float = 0.0,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "`d_model` should be dividable by `n_head`"

        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout_p=attn_dropout_p)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout_p = out_dropout_p

    def forward(
        self,
        qry: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W_q(qry)
        k = self.W_k(key)
        v = self.W_v(val)

        q = q.view(qry.size(0), qry.size(1), self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(key.size(0), key.size(1), self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(val.size(0), val.size(1), self.n_head, self.head_dim).transpose(1, 2)

        attn_feat, attn_weight = self.attention(q, k, v, attn_mask)
        # concatinate all features
        attn_feat = attn_feat.transpose(1, 2).reshape(qry.size(0), qry.size(1), self.n_head * self.head_dim)
        
        output = self.W_o(attn_feat)
        if self.dropout_p > 0.0:
            output = F.dropout(output, p=self.dropout_p, training=self.training)
        return output, attn_weight


class MHASubLayer(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        dropout_p: float = 0.0,
    ) -> None:
        super(MHASubLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mmha = MultiHeadAttention(
            n_head=n_head, d_model=d_model, attn_dropout_p=dropout_p, out_dropout_p=dropout_p
        )

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor.bool | None = None,
    ) -> torch.Tensor:
        res = self.norm(x)
        res , _ = self.mmha(res, res, res, attn_mask) # only get the attended features and discard attention weights
        output = x + res
        return output
