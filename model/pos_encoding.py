import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        max_context_len: int,
        d_model: int = 512,
        dropout_p: float = 0.0,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout_p = dropout_p

        pos_emb = torch.empty(max_context_len, d_model)
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02, a=-0.04, b=0.04)
        self.register_parameter("positional_embeddings", nn.Parameter(pos_emb))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        _, S, _ = embeddings.size()
        pos_emb = self.positional_embeddings[:S, :]
        output = embeddings + pos_emb.unsqueeze(0) # (B, S, D)
        if self.dropout_p > 0.0:
            output = F.dropout(output, p=self.dropout_p, training=self.training)
        return output


if __name__ == "__main__":
    from embedding import Embedding

    emb_layer = Embedding(30, d_model=64)
    pos_enc = PositionalEncoding(13, 64)

    test_in = torch.randint(30, size=(1, 13))

    test_out = pos_enc(emb_layer(test_in))
    print(test_out.size())
    print(test_out)
