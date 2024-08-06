from dataclasses import dataclass
from typing import OrderedDict
import torch
from architecture.rotary_emb import create_rotation_matrix

@dataclass
class Config:
    state_dict: OrderedDict[str, torch.Tensor]
    device: str
    n_embd: int
    n_head: int

    def __post_init__(self):
        self.embeddings = list(self.state_dict.values())[0]
        self.rotary_emb = create_rotation_matrix(self.embeddings)
