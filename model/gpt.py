import torch
import torch.nn as nn

from .embedding import Embedding
from .pos_encoding import PositionalEncoding
from .attention import MHASubLayer
from .feedforward import FFNSubLayer


class BabyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_context_len: int,
        n_layer: int = 6,
        n_head: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(BabyGPT, self).__init__()
        self.max_context_window = max_context_len

        self.emb_layer = Embedding(vocab_size, d_model=d_model)
        self.pos_encoding = PositionalEncoding(
            max_context_len, d_model=d_model, dropout_p=dropout_p
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layer):
            self.blocks.append(MHASubLayer(n_head=n_head, d_model=d_model, dropout_p=dropout_p))
            self.blocks.append(FFNSubLayer(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p))
        
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(
        self, 
        seq_ids: torch.Tensor, 
        attn_mask: torch.Tensor.bool
    ) -> torch.Tensor:
        x = self.pos_encoding(self.emb_layer(seq_ids))

        for i, lay in enumerate(self.blocks):
            if i % 2 == 0:
                x = lay(x, attn_mask)
            else:
                x = lay(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
