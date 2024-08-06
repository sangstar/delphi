import pytest
import torch
from torch import nn
from config import Config
from attention import GPTNeoXAttention


class MockConfig(Config):
    def __init__(self, state_dict, device, n_embd):
        super().__init__(state_dict, device, n_embd)
        self.rotary_emb = torch.randn(n_embd, n_embd)


@pytest.fixture
def mock_config():
    state_dict = {
        "gpt_neox.layers.0.attention.query_key_value.weight": torch.randn(3 * 128, 128),
        "gpt_neox.layers.0.attention.query_key_value.bias": torch.randn(3 * 128),
    }
    return MockConfig(state_dict, "cpu", 128)


def test_initialization(mock_config):
    attention = GPTNeoXAttention(mock_config, "gpt_neox.layers.0.attention.query_key_value")
    assert attention.config == mock_config
    assert isinstance(attention.attn, nn.Linear)
    assert attention.attn.weight.shape == (3 * mock_config.n_embd, mock_config.n_embd)
    assert attention.attn.bias.shape == (3 * mock_config.n_embd,)


def test_load_weights(mock_config):
    weight, bias = GPTNeoXAttention.load_weights(mock_config, "gpt_neox.layers.0.attention.query_key_value")
    assert weight.shape == (3 * mock_config.n_embd, mock_config.n_embd)
    assert bias.shape == (3 * mock_config.n_embd,)
    assert torch.equal(weight, mock_config.state_dict["gpt_neox.layers.0.attention.query_key_value.weight"])
    assert torch.equal(bias, mock_config.state_dict["gpt_neox.layers.0.attention.query_key_value.bias"])


def test_forward_pass(mock_config):
    attention = GPTNeoXAttention(mock_config, "gpt_neox.layers.0.attention.query_key_value")
    x = torch.randn(1, 10, mock_config.n_embd)  # Batch size 1, sequence length 10
    y = attention.forward(x)
    assert y.shape == (1, 10, mock_config.n_embd)
    assert not torch.isnan(y).any(), "Output contains NaNs"
    assert not torch.isinf(y).any(), "Output contains Infs"


if __name__ == "__main__":
    pytest.main()
