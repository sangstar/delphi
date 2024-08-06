from torch import nn
from config import Config


class GPTNeoXAttention(nn.Module):

    def __init__(self, config: Config, name):
        super().__init__()
        self.config = config
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn.weight, self.attn.bias = self.load_weights(config, name)

    def forward(self, x):
        q, k, v = self.attn(x).split(self.config.n_embd, dim=-1)
        q = q @ self.config.rotary_emb
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        return y

    @staticmethod
    def load_weights(config: Config, name: str):
        weight = nn.Parameter(config.state_dict[f"{name}.weight"])
        bias = nn.Parameter(config.state_dict[f"{name}.bias"])
        return weight, bias
