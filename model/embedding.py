import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.emb_layer = nn.Embedding(vocab_size, d_model)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.emb_layer(token_ids)
        return output
    

if __name__ == "__main__":
    emb_layer = Embedding(30, d_model=64)

    test_in = torch.randint(30, size=(1, 3))

    test_out = emb_layer(test_in)

    print(test_out.size())
    print(test_out)
