import pytest
import torch
from torch import nn
import architecture as arch
from gpt_neox import GPTNeoXModel

model_name = "EleutherAI/pythia-14m"

@pytest.fixture
def sample_config():
    model = GPTNeoXModel.from_pretrained(model_name)
    config = model.config
    config.device = "cpu"
    config.n_embd = 128
    return config

@pytest.fixture
def model(sample_config):
    return GPTNeoXModel(sample_config)

def test_initialization(model, sample_config):
    assert isinstance(model.transformer, nn.ModuleDict)
    assert isinstance(model.transformer.emb_in, arch.Embedding)
    assert isinstance(model.transformer.h, nn.ModuleList)
    assert len(model.transformer.h) == len(sample_config.state_dict) - 2
    assert isinstance(model.transformer.ln_f, arch.LayerNorm)
    assert isinstance(model.transformer.emb_out, arch.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.out_features == sample_config.vocab_size

def test_from_pretrained():
    model = GPTNeoXModel.from_pretrained(model_name)
    assert isinstance(model, GPTNeoXModel)
    assert model.config.device == "cpu"
    assert model.config.n_embd == 128

def test_forward(model, sample_config):
    input_tensor = torch.randint(0, 1000, (1, 10))  # Dummy input tensor
    output = model.forward(input_tensor)
    assert output.shape == (1, 1, sample_config.vocab_size)

