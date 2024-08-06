from torch import nn
from config import Config

# TODO: Transformer layers need to be executed in parallel
class GPTNeoXAttention(nn.Module):

    def __init__(self, config: Config, name):
        super().__init__()
        self.config = config

        # Produces 3 vectors per token vector: q, k, v with same dim as x
        # Therefore must have a 3 * n_embd for output dim
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn.weight, self.attn.bias = self.load_weights(config, name)

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
        return y.transpose(1, 2).contiguous().view(
            batch_size,
            seq_length,
            n_embd)

    @staticmethod
    def load_weights(config: Config, name: str):
        weight = nn.Parameter(config.state_dict[f"{name}.weight"])
        bias = nn.Parameter(config.state_dict[f"{name}.bias"])
        return weight, bias

# class MLP(nn.Module):
#
#
#
# class TransformerLayer(nn.Module)
# Combines the LayerNorm, MLP, and Attention layers
