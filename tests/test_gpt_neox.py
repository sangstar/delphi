import pytest
import torch
from torch import nn
import architecture as arch
from gpt_neox import GPTNeoXModel
from transformers import AutoModelForCausalLM, AutoConfig

model_name = "EleutherAI/pythia-14m"
gpt_neox_model = GPTNeoXModel.from_pretrained(model_name)
gpt_neox_config = gpt_neox_model.config
# TODO: Parametrize these tests for GPU device and different GPT-NeoX models

@pytest.fixture
def sample_config():
    return gpt_neox_config

@pytest.fixture
def model(sample_config):
    return gpt_neox_model

def test_initialization(model, sample_config):
    assert isinstance(model.transformer, nn.ModuleDict)
    assert isinstance(model.transformer.emb_in, arch.Embedding)
    assert isinstance(model.transformer.h, nn.ModuleList)
    assert len(model.transformer.h) == sample_config.num_hidden_layers
    assert isinstance(model.transformer.ln_f, arch.LayerNorm)
    assert isinstance(model.transformer.emb_out, arch.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.out_features == sample_config.vocab_size

def test_from_pretrained():
    model = GPTNeoXModel.from_pretrained("EleutherAI/pythia-14m")
    assert isinstance(model, GPTNeoXModel)
    assert model.config.device == "cpu"
    assert model.config.n_embd == 128

def test_forward(model, sample_config):
    input_tensor = torch.randint(0, 1000, (1, 10))  # Dummy input tensor
    output = model.forward(input_tensor)
    assert output.shape == (1, 1, sample_config.vocab_size)


def test_generate(model, sample_config):
    prompt = "Hi, my name is "
    model.generate(prompt, max_length=10)

def test_load_weights(model, sample_config):
    hf_model, hf_config = AutoModelForCausalLM.from_pretrained(model_name), AutoConfig.from_pretrained(model_name)
    for layers in hf_model.state_dict():
        assert(torch.allclose(model.config.state_dict[layers],hf_model.state_dict()[layers]))