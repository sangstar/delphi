import torch.nn as nn
import architecture as arch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from config import Config

class GPTNeoXModel:

    def __init__(self, config):
        self.config = config

        self.num_transformer_layers = self.config.num_hidden_layers

        # Load model weights in to arch modules
        self.transformer = nn.ModuleDict(dict(
            emb_in=arch.Embedding(config, "gpt_neox.embed_in"),
            h=nn.ModuleList((arch.TransformerBlock(config, i)) for i in
                            range(self.num_transformer_layers)),
            ln_f=arch.LayerNorm(config, "gpt_neox.final_layer_norm"),
            emb_out=arch.Embedding(config, "embed_out")
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # TODO: Allow this to be loaded locally/over HTTP with tensorizer, use
    #       no_init_or_tensor to avoid double-loading
    @classmethod
    def from_pretrained(cls, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        config = Config(model.state_dict(), model.config, dtype=model.dtype)
        config.state_dict = model.state_dict()
        return cls(config)

    def forward(self, x):
        x = self.transformer.emb_in(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])

        return logits

