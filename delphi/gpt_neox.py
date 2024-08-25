"""
An implementation of GPT-NeoX.

References:
    1) HuggingFace's rotary embeddings implementation
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L573

    2) The official GPT-NeoX paper
    https://arxiv.org/pdf/2204.06745v1

    3) Andrej Karpathy's NanoGPT (inspiration, generation method)
    https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GPTNeoXConfig:
    hidden_size: int = None
    num_attention_heads: int = None
    intermediate_size: int = None
    vocab_size: int = None
    num_hidden_layers: int = None
    rotary_emb_base: int = None
    rotary_pct: int = None
    max_position_embeddings: int = None
    device: torch.device = None

    debug: bool = False
    use_kv_cache: bool = True

    def __post_init__(self):
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_default_device(self.device)

        if self.use_kv_cache:
            self._kv_cache = {}
            self._kv_hashes = {}

    @classmethod
    def from_hf(cls, config: transformers.GPTNeoXConfig):
        return GPTNeoXConfig(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.vocab_size,
            config.num_hidden_layers,
            config.rotary_emb_base,
            config.rotary_pct,
            config.max_position_embeddings,
        )

    def _update_kv_cache(self, update_dict: dict):
        self._kv_cache.update(update_dict)
        if self.debug:
            idx = list(update_dict.keys())[0]
            if idx in self._kv_hashes.keys():
                self._kv_hashes.pop(idx)

    def _validate_kv_cache_checksums(self):
        hash_tensor = lambda x: hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
        for idx, kv in self._kv_cache.items():
            checksum = ""
            for tensor in kv:
                checksum += hash_tensor(tensor[:, :, :])
            combined_checksum = hashlib.sha256(checksum.encode("utf-8")).hexdigest()
            if idx not in self._kv_hashes.keys():
                self._kv_hashes.update({idx: combined_checksum})
            else:
                assert combined_checksum == self._kv_hashes[idx]

    def _retrieve_from_kv_cache(self, idx):
        if self.debug:
            self._validate_kv_cache_checksums()
        return self._kv_cache[idx]


# From transformers.models.gpt_neox.modeling_gpt_neox
class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
        )

    def _set_cos_sin_cache(self, seq_len, device):
        # Sequences this size or less can use the same cached sin/cos
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: GPTNeoXConfig, idx):
        super().__init__()
        self.config = config
        self.query_key_value = nn.Linear(
            self.config.hidden_size, 3 * self.config.hidden_size
        )
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.n_embd = self.config.hidden_size
        self.n_head = self.config.num_attention_heads

        # Rotary Embeddings related parameters
        self.head_size = self.n_embd // self.n_head  # Per-head size
        self.rotary_ndims = int(self.head_size * self.config.rotary_pct)
        self.rotary_emb = GPTNeoXRotaryEmbedding(
            self.rotary_ndims,
            self.config.max_position_embeddings,
            base=self.config.rotary_emb_base,
        )
        self.idx = idx

        self.query_key_value.requires_grad_(False)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()
        q, k, v = self._attention_projections_and_rope(x)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        y = self.dense(y)
        return y

    def get_q_matrix(self, x):
        weight = self.query_key_value.weight
        bias = self.query_key_value.bias

        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)

        q = F.linear(x, q_weight, q_bias)

        return q

    def _to_qkv(self, x):
        batch_size, seq_len, n_embd = x.size()
        if not self.config.use_kv_cache or self.idx not in self.config._kv_cache.keys():
            qkv = self.query_key_value(x)
            _, k_cache, v_cache = qkv.split(self.n_embd, dim=-1)
            self.config._update_kv_cache({self.idx: (k_cache, v_cache)})

            qkv = qkv.view(batch_size, seq_len, self.n_head, 3 * self.head_size)
            q, k, v = qkv.split(self.head_size, dim=-1)

            return q, k, v
        else:
            q = self.get_q_matrix(x)

            cached_k, cached_v = self.config._retrieve_from_kv_cache(self.idx)

            tokens_to_calculate = seq_len - cached_k.size(1)

            # This technically calculates the q vector of the last token
            # extra time(s)
            next_tok_qkv = self.query_key_value(x[:, -tokens_to_calculate:, :])
            _, k_next, v_next = next_tok_qkv.split(self.n_embd, dim=-1)

            # Concatenate cached K, V with new K, V
            k_new = torch.cat((cached_k, k_next), dim=1)
            v_new = torch.cat((cached_v, v_next), dim=1)

            self.config._update_kv_cache({self.idx: (k_new, v_new)})

            qkv = torch.cat((q, k_new, v_new), dim=-1)
            qkv = qkv.view(batch_size, seq_len, self.n_head, 3 * self.head_size)
            q, k, v = qkv.split(self.head_size, dim=-1)
            return q, k, v

    def _split_to_qkv_vectors_by_head(self, x):
        batch_size, seq_len, _ = x.size()
        q, k, v = self._to_qkv(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        return q, k, v

    def _get_rotary_params(self, q, k, v):
        q_rot, q_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
        k_rot, k_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
        cos, sin = self.rotary_emb(v, seq_len=k.size(-2))
        return (q_rot, q_pass), (k_rot, k_pass), cos, sin

    def _apply_rotary_embeddings_and_concatenate(self, q_params, k_params, cos, sin):
        q_rot, q_pass = q_params
        k_rot, k_pass = k_params
        position_ids = torch.arange(0, k_pass.size(-2)).unsqueeze(0)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)

        q, k = torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def _attention_projections_and_rope(self, x):
        q, k, v = self._split_to_qkv_vectors_by_head(x)
        q_params, k_params, cos, sin = self._get_rotary_params(q, k, v)
        q, k = self._apply_rotary_embeddings_and_concatenate(
            q_params, k_params, cos, sin
        )
        return q, k, v


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        return x


class Block(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.idx = idx
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.attention = GPTNeoXAttention(config, self.idx)
        self.mlp = FeedForward(config)

    def forward(self, x):
        attn_output = self.attention(self.input_layernorm(x))
        mlp_output = self.mlp(self.post_attention_layernorm(x))
        x = mlp_output + attn_output + x
        return x


class GPTNeoX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                embed_in=nn.Embedding(config.vocab_size, config.hidden_size),
                layers=nn.ModuleList(
                    Block(config, i) for i in range(config.num_hidden_layers)
                ),
                final_layer_norm=nn.LayerNorm(config.hidden_size),
            )
        )

        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_out.requires_grad_(False)

        self.transformer.requires_grad_(False)

    def forward(self, x):
        x = self.transformer.embed_in(x)
        for module in self.transformer.layers:
            x = module(x)
        x = self.transformer.final_layer_norm(x)
        logits = self.embed_out(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_ref, config: GPTNeoXConfig = None):
        hf_model = AutoModelForCausalLM.from_pretrained(model_ref)
        hf_config = hf_model.config
        assert (
            "GPTNeoXForCausalLM" in hf_config.architectures
        ), "Must be a GPTNeoX model"

        model = GPTNeoX(
            GPTNeoXConfig(
                hf_config.hidden_size,
                hf_config.num_attention_heads,
                hf_config.intermediate_size,
                hf_config.vocab_size,
                hf_config.num_hidden_layers,
                hf_config.rotary_emb_base,
                hf_config.rotary_pct,
                hf_config.max_position_embeddings,
            )
        )

        # Add non-architectural params
        if config:
            for key, value in config.__dict__.items():
                if value is not None:
                    setattr(model.config, key, value)

        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        hf_sd = hf_model.state_dict()

        model.state_dict = OrderedDict()
        model.transformer.embed_in.weight.copy_(hf_sd["gpt_neox.embed_in.weight"])

        model.state_dict["gpt_neox.embed_in.weight"] = hf_sd["gpt_neox.embed_in.weight"]

        # Set weights and biases for each layer
        for i in range(model.config.num_hidden_layers):
            layer = model.transformer.layers[i]
            for module_name, module in layer.named_children():
                if hasattr(module, "weight") and hasattr(module, "bias"):
                    weight, bias = (
                        f"gpt_neox.layers.{i}.{module_name}.weight",
                        f"gpt_neox.layers.{i}.{module_name}.bias",
                    )
                    module.weight.copy_(hf_sd[weight])
                    module.bias.copy_(hf_sd[bias])
                    model.state_dict[weight] = hf_sd[weight]
                    model.state_dict[bias] = hf_sd[bias]
                else:
                    for submodule_name, submodule in module.named_children():
                        if submodule_name not in ["gelu", "rotary_emb"]:
                            weight, bias = (
                                f"gpt_neox.layers.{i}.{module_name}.{submodule_name}.weight",
                                f"gpt_neox.layers.{i}.{module_name}.{submodule_name}.bias",
                            )
                            submodule.weight.copy_(hf_sd[weight])
                            submodule.bias.copy_(hf_sd[bias])
                            model.state_dict[weight] = hf_sd[weight]
                            model.state_dict[bias] = hf_sd[bias]

        # Set final layer norm weights and biases
        model.transformer.final_layer_norm.weight.copy_(
            hf_sd["gpt_neox.final_layer_norm.weight"]
        )
        model.transformer.final_layer_norm.bias.copy_(
            hf_sd["gpt_neox.final_layer_norm.bias"]
        )

        # Set final embed_out weight
        model.embed_out.weight.copy_(hf_sd["embed_out.weight"])

        model.tokenizer = tokenizer

        for key, value in model.state_dict.items():
            assert torch.allclose(model.state_dict[key], hf_sd[key], atol=1e-10)

        return model

    @torch.no_grad()
    def generate(
        self, prompt, max_new_tokens, temperature=1.0, top_k=None, stream=False
    ):
        idx = self.tokenizer.encode(prompt, return_tensors="pt")
        for _ in range(max_new_tokens):

            if idx.size()[-1] > self.config.max_position_embeddings:
                raise ValueError("Prompt exceeds block size")

            logits = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == self.tokenizer.eos_token_id:
                return self.tokenizer.decode(idx[0], skip_special_tokens=True)
            if stream:
                yield idx_next

        return self.tokenizer.decode(idx[0], skip_special_tokens=True)
