import pytest
import torch
from torch import nn
from architecture import GPTNeoXAttention
from config import Config
from transformers import AutoModelForCausalLM, AutoConfig

class MockConfig(Config):
    def __init__(self, state_dict, hf_config):
        super().__init__(state_dict, hf_config)


@pytest.fixture
def mock_config():
    hf_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    return MockConfig(state_dict=hf_model.state_dict(), hf_config=config)

def test_initialization(mock_config):
    attention = GPTNeoXAttention(mock_config, "gpt_neox.layers.0.attention.query_key_value")
    assert attention.config == mock_config
    assert isinstance(attention.attn, nn.Linear)
    assert attention.attn.weight.shape == (3 * mock_config.n_embd, mock_config.n_embd)
    assert attention.attn.bias.shape == (3 * mock_config.n_embd,)

def test_forward_pass(mock_config):
    attention = GPTNeoXAttention(mock_config, "gpt_neox.layers.0.attention.query_key_value")
    x = torch.randn(1, 10, mock_config.n_embd)  # Batch size 1, sequence length 10
    y = attention.forward(x)
    assert y.shape == (1, 10, mock_config.n_embd)
    assert not torch.isnan(y).any(), "Output contains NaNs"
    assert not torch.isinf(y).any(), "Output contains Infs"

if __name__ == "__main__":
    pytest.main()
