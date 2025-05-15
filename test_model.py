import torch

from model.gpt import BabyGPT


def main():
    gpt = BabyGPT(
        vocab_size=39, 
        max_context_len=32,
        n_layer=6,
        n_head = 8,
        d_model=256,
        d_ff=256*4,
        dropout_p=0.1,
    )

    test_in = torch.randint(39, size=(4, 32))
    causal_mask = torch.ones(4, 32, 32)
    causal_mask = torch.tril(causal_mask)
    causal_mask = causal_mask.to(dtype=torch.bool)

    test_out = gpt(test_in, causal_mask)

    print(test_out.size())
    print(test_out)


if __name__ == "__main__":
    main()
