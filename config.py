from dataclasses import dataclass
from typing import OrderedDict
import torch
from rotary_emb import RotaryEmbeddings
from transformers import GPTNeoXConfig

@dataclass
class Config:
    state_dict: OrderedDict[str, torch.Tensor]
    hf_config: GPTNeoXConfig
    device: str = "cpu"
    dtype: torch.dtype = None

    def __post_init__(self):
        self.dtype = self.hf_config.torch_dtype if self.dtype is None else self.dtype

        torch.set_default_dtype(self.dtype)

        self.embeddings = list(self.state_dict.values())[0]

        rotary_emb = RotaryEmbeddings(self)
        self.rotary_emb = rotary_emb(self.embeddings)

        self.n_head = self.hf_config.num_attention_heads
        self.vocab_size = self.hf_config.vocab_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.n_embd = self.hf_config.hidden_size

