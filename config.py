from dataclasses import dataclass
from typing import OrderedDict
import torch
from rotary_emb import create_rotation_matrix
from transformers import GPTNeoXConfig

@dataclass
class Config:
    state_dict: OrderedDict[str, torch.Tensor]
    hf_config: GPTNeoXConfig
    device: str = "cpu"

    def __post_init__(self):
        self.embeddings = list(self.state_dict.values())[0]
        self.rotary_emb = create_rotation_matrix(self.embeddings)
        self.n_head = self.hf_config.num_attention_heads
        self.vocab_size = self.hf_config.vocab_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.n_embd = self.hf_config.hidden_size

