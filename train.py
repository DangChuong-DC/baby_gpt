from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tokenizer import CharTokenizer
from dataset import TinyShakespeare, collate_fn
from model.gpt import BabyGPT
from model.config import MODEL_CONFIG


def get_causal_mask(inp: torch.Tensor, device: str) -> torch.Tensor.bool:
    causal_mask = torch.ones(inp.size(0), inp.size(1), inp.size(1))
    causal_mask = torch.tril(causal_mask)
    causal_mask = causal_mask.to(dtype=torch.bool, device=device)
    return causal_mask


def train(
    epoch: int, 
    model: nn.Module, 
    dataloader: DataLoader, 
    device: str, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    log_iter: int = 900,
) -> None:
    model.train()
    train_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        inp = batch["inputs"].to(device=device)
        gt = batch["labels"].to(device=device)

        optimizer.zero_grad()
        # create causal mask
        causal_mask = get_causal_mask(inp, device)

        logits = model(inp, causal_mask)
        assert torch.isfinite(logits).all(), "NaN in logits!"
        B, S, D = logits.size()
        flatten_logits = logits.view(B * S, D)
        flatten_gt = gt.view(B * S)
        loss = criterion(flatten_logits, flatten_gt)
        assert torch.isfinite(loss).all(), "loss became NaN!"
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()

        if i % log_iter == 0:
            print(f"[Iter {i} --- loss: {loss.item()}]")
    
    avg_train_loss = train_loss / len(dataloader)
    print(f"> 🎓 [Epoch {epoch}] train loss: {avg_train_loss:.9f} <")


def eval(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> None:
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inp = batch["inputs"].to(device=device)
            gt = batch["labels"].to(device=device)

            mask = get_causal_mask(inp, device)
            logits = model(inp, mask)
            B, S, D = logits.size()
            flatten_logits = logits.view(B * S, D)
            flatten_gt = gt.view(B * S)
            loss = criterion(flatten_logits, flatten_gt)

            eval_loss += loss.item()
    
    avg_eval_loss = eval_loss / len(dataloader)
    print(f">>> 🔍 [Eval epoch {epoch}] eval loss: {avg_eval_loss:.9f} <<<")


def main():
    # ___ Training hyperparameters ___
    num_epoch = 9 # 13
    max_learning_rate = 5e-4
    min_learning_rate = 6e-5
    batch_size = 512 # 256
    logging_iter = 900
    model_save_dir = f"/home/dc/self_studies/baby_gpt/checkpoints/"
    model_id = "20250620_002"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # create model save directory if not exists
    model_save_dir = Path(model_save_dir) / model_id
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created model save directory: {model_save_dir}")

    # ___ Data preparation ___
    ### Uncommented data_path lines below for Vietnamese dataset
    tokenizer = CharTokenizer("en")  # "en" for English, "vi" for Vietnamese
    dset_train = TinyShakespeare(
        tokenizer, "train", max_context_len=MODEL_CONFIG["max_context_window"], train_val_ratio=0.9,
        # data_path="/home/dc/self_studies/baby_gpt/data/xuandieutho.txt",
        data_path="/DATA01/dc/datasets/tiny_shakespeare/data.txt",
    )
    dset_val = TinyShakespeare(
        tokenizer, "val", max_context_len=MODEL_CONFIG["max_context_window"], train_val_ratio=0.9,
        # data_path="/home/dc/self_studies/baby_gpt/data/xuandieutho.txt",
        data_path="/DATA01/dc/datasets/tiny_shakespeare/data.txt",
    )
    
    train_dataloader = DataLoader(
        dset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_dataloader = DataLoader(
        dset_val, batch_size=batch_size*2, shuffle=True, collate_fn=collate_fn, num_workers=2
    )


    # Model
    gpt = BabyGPT(
        vocab_size=tokenizer.get_vocab_size(),
        max_context_len=MODEL_CONFIG["max_context_window"],
        n_layer=MODEL_CONFIG["num_of_layer"],
        n_head=MODEL_CONFIG["num_of_attn_head"],
        d_model=MODEL_CONFIG["model_dim"],
        d_ff=MODEL_CONFIG["feedforward_dim"],
        dropout_p=MODEL_CONFIG["dropout_rate"],
    )
    gpt.to(device=device)

    optimizer = AdamW(
        gpt.parameters(), lr=max_learning_rate, betas=(0.9, 0.95)
    )
    total_steps = num_epoch * len(train_dataloader)
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps, min_learning_rate)
    criterion = nn.CrossEntropyLoss()

    for e in range(num_epoch):
        eval(e, gpt, val_dataloader, device, criterion)
        train(
            e, gpt, train_dataloader, device, optimizer, criterion, lr_scheduler, log_iter=logging_iter
        )

        # after training loop
        file_name = f"babygpt_shakespeare_weights_{e:03d}.pt"
        model_save_path = Path(model_save_dir) / file_name
        torch.save(gpt.state_dict(), model_save_path)
        print(f"✅ Saved model weights to {model_save_path}")


if __name__ == "__main__":
    main()
