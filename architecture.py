from torch import nn
from config import Config
from torch.nn import functional as F


def load_weights(config: Config, name: str):
    weight = nn.Parameter(config.state_dict[f"{name}.weight"]) \
        if f"{name}.weight" in config.state_dict else None
    bias = nn.Parameter(config.state_dict[f"{name}.bias"]) \
        if f"{name}.bias" in config.state_dict else None
    return weight, bias


class Embedding(nn.Embedding):

    def __init__(self, config: Config, name):
        super().__init__(config.vocab_size, config.n_embd)
        self.config = config
        self.weight, _ = load_weights(config, name)


class GPTNeoXAttention(nn.Module):

    def __init__(self, config: Config, name):
        super().__init__()
        self.config = config

        # Produces 3 vectors per token vector: q, k, v with same dim as x
        # Therefore must have a 3 * n_embd for output dim
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn.weight, self.attn.bias = load_weights(config, name)

        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.dense.weight, self.dense.bias = load_weights(
            config,
            f"{name[:-len('query_key_value')]}dense"
        )

    def forward(self, x):
        batch_size, seq_length, n_embd = x.shape
        nh = self.config.n_head
        hs = n_embd // nh

        q, k, v = self.attn(x).split(n_embd, dim=-1)

        # q vector is multiplied by the rotation matrix as per the gpt-neox paper,
        # so will stand this in for q
        q = q @ self.config.rotary_emb

        # Reshape and transpose to get (B, nh, T, hs)
        q = q.view(batch_size, seq_length, nh, hs).transpose(1, 2)
        k = k.view(batch_size, seq_length, nh, hs).transpose(1, 2)
        v = v.view(batch_size, seq_length, nh, hs).transpose(1, 2)

        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        y = y.transpose(1, 2).contiguous().view(
            batch_size,
            seq_length,
            n_embd)
        return self.dense(y)


class LayerNorm(nn.Module):

    def __init__(self, config, name):
        super().__init__()
        self.weight, self.bias = load_weights(config, name)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_fc.weight, self.c_fc.bias = load_weights(
            config,
            f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.weight, self.c_proj.bias = load_weights(
            config,
            f"gpt_neox.layers.{layer_num}.mlp.dense_4h_to_h"
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: Config, layer_num: int):
        super().__init__()

        ln1_name = f"gpt_neox.layers.{layer_num}.input_layernorm"
        ln2_name = f"gpt_neox.layers.{layer_num}.post_attention_layernorm"
        attn_name = f"gpt_neox.layers.{layer_num}.attention.query_key_value"
        self.ln_1 = LayerNorm(config, ln1_name)
        self.attn = GPTNeoXAttention(config, attn_name)
        self.ln_2 = LayerNorm(config, ln2_name)
        self.mlp = MLP(config, layer_num)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
