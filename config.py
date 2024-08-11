from dataclasses import dataclass
from typing import OrderedDict
import torch
from transformers import GPTNeoXConfig

@dataclass
class Config:
    state_dict: OrderedDict[str, torch.Tensor]
    hf_config: GPTNeoXConfig = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.dtype = self.hf_config.torch_dtype if self.dtype is None else self.dtype

        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.device)

        self.seq_length = None

        self.embeddings = list(self.state_dict.values())[0]

        self.rotary_angle = self.hf_config.rotary_emb_base

        self.n_head = self.hf_config.num_attention_heads
        self.vocab_size = self.hf_config.vocab_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        self.intermediate_size = self.hf_config.intermediate_size
        self.n_embd = self.hf_config.hidden_size


