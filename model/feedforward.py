import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(FeedForward, self).__init__()
        self.dropout_p = dropout_p

        self.W1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.W1(x))
        x = self.W2(x)
        if self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
    

class FFNSubLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(FFNSubLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout_p=dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.norm(x)
        res = self.ffn(res)
        output = x + res
        return output
