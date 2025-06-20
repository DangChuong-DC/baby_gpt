from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0) -> None:
        super(SimpleSelfAttention, self).__init__()
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
            attn_bias.masked_fill_(attn_mask.logical_not(), -math.inf) # (B, S_q, S_k)
            if qry.ndim == 4: # (B, H, S, D)
                attn_bias.unsqueeze_(1)

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

        self.attention = SimpleSelfAttention(dropout_p=attn_dropout_p)

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


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-head causal attention layer.
    This layer is used in the decoder part of the Transformer model.
    This implementation is heavily inspired from implementation of Karpathy's nanoGPT
    """
    def __init__(
        self,
        n_head: int,
        d_model: int,
        attn_dropout_p: float = 0.1,
        out_dropout_p: float = 0.1,
        flash_attn: bool = True,
    ) -> None:
        super(MultiHeadCausalAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attn_dropout_p = attn_dropout_p
        self.out_dropout_p = out_dropout_p

        self.attn_proj = nn.Linear(d_model, 3 * d_model, bias=False) # from input project to q, k, v
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.use_flash_attn = False
        if flash_attn and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("!!! Will use Flash Attention for MultiHeadCausalAttention")
            self.use_flash_attn = True

    def _attention(
        self,
        qry: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool
    ) -> torch.Tensor:
        attn_bias = None
        if attn_mask is not None:
            B = qry.size(0)
            S_q, S_k = qry.size(-2), key.size(-2)
            attn_bias = torch.zeros((B, S_q, S_k), dtype=qry.dtype, device=qry.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), -math.inf) # (B, S_q, S_k)
            attn_bias.unsqueeze_(1) # (B, 1, S_q, S_k)

        scaled_dot_prod = torch.matmul(qry, key.transpose(-1, -2)) / math.sqrt(key.size(-1))
        if attn_bias is not None:
            scaled_dot_prod += attn_bias
        attn_weights = F.softmax(scaled_dot_prod, dim=-1)
        if self.attn_dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, self.attn_dropout_p, training=self.training)
        output = torch.matmul(attn_weights, val)
        return output

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor.bool) -> torch.Tensor:
        x = self.attn_proj(x) # (B, S, D*3)

        qry, key, val = x.split(self.d_model, dim=-1) # (B, S, D)

        qry = qry.view(
            qry.size(0), qry.size(1), self.n_head, self.d_model // self.n_head
        ).transpose(1, 2) # (B, n_head, S, dim)
        key = key.view(
            key.size(0), key.size(1), self.n_head, self.d_model // self.n_head
        ).transpose(1, 2)
        val = val.view(
            val.size(0), val.size(1), self.n_head, self.d_model // self.n_head
        ).transpose(1, 2)

        if self.use_flash_attn:
            attn_feat = F.scaled_dot_product_attention(
                qry, key, val, dropout_p=self.attn_dropout_p, is_causal=True
            )
            # not parsing attn_mask but using native implementation from Pytorch by set `is_causal` instead
        else:
            attn_feat = self._attention(qry, key, val, attn_mask=attn_mask) # (B, n_head, S, dim)
        attn_feat = attn_feat.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        output = self.out_proj(attn_feat)
        if self.out_dropout_p > 0.0:
            output = F.dropout(output, p=self.out_dropout_p, training=self.training)
        return output


class MHCASubLayer(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        dropout_p: float = 0.0,
    ) -> None:
        super(MHCASubLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mmha = MultiHeadCausalAttention(
            n_head=n_head, d_model=d_model, attn_dropout_p=dropout_p, 
            out_dropout_p=dropout_p, flash_attn=True
        )

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor.bool | None = None,
    ) -> torch.Tensor:
        res = self.norm(x)
        res = self.mmha(res, attn_mask)
        output = x + res
        return output
