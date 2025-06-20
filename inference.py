import time

import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from model.gpt import BabyGPT
from model.config import MODEL_CONFIG

MODEL_PATH = "/home/dc/self_studies/baby_gpt/checkpoints/20250620_002/babygpt_shakespeare_weights_008.pt"


def sample_with_temperature(logits: torch.Tensor, temperature: float):
    # logits: shape (vocab_size,)
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer("en")  # "en" for English, "vi" for Vietnamese

    gpt = BabyGPT(
        vocab_size=tokenizer.get_vocab_size(),
        max_context_len=MODEL_CONFIG["max_context_window"],
        n_layer=MODEL_CONFIG["num_of_layer"],
        n_head=MODEL_CONFIG["num_of_attn_head"],
        d_model=MODEL_CONFIG["model_dim"],
        d_ff=MODEL_CONFIG["feedforward_dim"],
        dropout_p=MODEL_CONFIG["dropout_rate"],
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gpt.load_state_dict(checkpoint)

    gpt = gpt.to(device=device)

    generated_text = "\n"
    while True:
        token_ids = tokenizer.encode(generated_text).to(device=device)

        # causal = torch.ones(1, token_ids.size(1), token_ids.size(1))
        # causal = torch.tril(causal).to(dtype=torch.bool, device=device)

        # logits = gpt(token_ids, causal)
        logits = gpt(token_ids, None) # remove causal if you use flash attention
        logits = logits[:, -1, :]
        new_token_id = sample_with_temperature(logits, 0.7)
        new_char = tokenizer.decode(new_token_id.cpu())
        generated_text += new_char
        generated_text = generated_text[-gpt.max_context_window:]

        print(generated_text[-1], end="", flush=True)
        # print("...💬")

        time.sleep(0.01)


if __name__ == "__main__":
    main()
