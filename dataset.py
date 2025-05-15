from __future__ import annotations

from typing import Any, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TinyShakespeare(Dataset):
    def __init__(
        self,
        tokenizer: nn.Module,
        split: str,
        max_context_len: int = 32,
        train_val_ratio: float = 0.9,
        data_path: str | Path = "/DATA01/dc/datasets/tiny_shakespeare/data.txt",
    ) -> None:
        super(TinyShakespeare, self).__init__()
        self.data_path = Path(data_path)
        self.split_ratio = train_val_ratio
        self.max_context_len = max_context_len
        
        data = self._load_raw_data(data_path)

        data_len = len(data)
        split_idx = int(self.split_ratio * data_len)

        
        if split == "train":
            self.data = data[:split_idx]
        elif split == "val":
            self.data = data[split_idx:]
        else:
            raise NotImplementedError(f"Do not support split: {split}")

        assert len(self.data) >= self.max_context_len, f"The split is too small, please change `train_val_ratio`"
        self.data_len = len(self.data) - self.max_context_len
        self.tokenizer = tokenizer

    def _load_raw_data(self, data_path: Path) -> str:
        with open(data_path, "r") as _f:
            data = _f.read()
        return data
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw_input = self.data[index:index + self.max_context_len]
        input = self.tokenizer.encode(raw_input) # (1, S)
        # ground truth shift right
        raw_gt = self.data[index + 1: index + self.max_context_len + 1]
        gt = self.tokenizer.encode(raw_gt) # (1, S)
        assert input.size(-1) == gt.size(-1), f"Expected length of input and \
            groundtruth to be equal; got {input.size(-1)} and {gt.size(-1)}"
        return {
            "input": input,
            "groundtruth": gt,
        }
    
    def __len__(self) -> int:
        return self.data_len
    

def collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inp = torch.cat([x["input"] for x in batch], dim=0)
    gt = torch.cat([x["groundtruth"] for x in batch], dim=0)
    return {
        "inputs": inp,
        "labels": gt,
    }


if __name__ == "__main__":
    from tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    dataset = TinyShakespeare(
        tokenizer,
        "train",
        max_context_len=8,
        train_val_ratio=0.3,
    )

    # print(len(dataset.data))
    dset_len = dataset.data_len
    
    """
    "ljdkladsk" -> len=9
     012345678
    => len - 4 = 5
    """
    

    test_output = dataset.__getitem__(dset_len - 1)
    print(test_output)
    
