import pytest
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, \
    apply_rotary_pos_emb
from transformers import AutoConfig, AutoModelForCausalLM
from gpt_neox import GPTNeoXAttention, GPTNeoXConfig, GPTNeoX
import torch

model_ref = "EleutherAI/pythia-14m"
ref_config = AutoConfig.from_pretrained(model_ref)

config = GPTNeoXConfig.from_hf(ref_config)
model = GPTNeoX.from_pretrained(model_ref)
ref_model = AutoModelForCausalLM.from_pretrained(model_ref)


@pytest.fixture
def example_input():
    # Small input tensor
    x = torch.randint(0, 100, (1, 10))
    example = model.transformer.embed_in(x)
    reference = ref_model.gpt_neox.embed_in(x)

    assert torch.allclose(example, reference,
                          atol=1e-6), \
        "Embedding layer outputs do not match"
    return example


def test_same_config():
    for param in config.__dict__.keys():
        model_param = getattr(config, param)
        ref_param = getattr(ref_config, param)
        assert model_param == ref_param


def test_same_weights():
    sd = model.state_dict
    hf_sd = ref_model.state_dict()
    for name, name_ref in zip(sd, hf_sd):
        assert torch.allclose(sd[name], hf_sd[name_ref],
                              atol=1e-8)

    from collections import OrderedDict
    model.state_dict = OrderedDict()

    assert torch.allclose(model.transformer.embed_in.weight,
                          hf_sd['gpt_neox.embed_in.weight'])

    # Set weights and biases for each layer
    for i in range(config.num_hidden_layers):
        layer = model.transformer.layers[i]
        for module_name, module in layer.named_children():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                weight, bias = f'gpt_neox.layers.{i}.{module_name}.weight', f'gpt_neox.layers.{i}.{module_name}.bias'
                assert torch.allclose(module.weight, hf_sd[weight], atol=1e-8)
                assert torch.allclose(module.bias, hf_sd[bias], atol=1e-8)
            else:
                for submodule_name, submodule in module.named_children():
                    if submodule_name not in ["gelu", "rotary_emb"]:
                        weight, bias = (
                            f'gpt_neox.layers.{i}.{module_name}.{submodule_name}.weight',
                            f'gpt_neox.layers.{i}.{module_name}.{submodule_name}.bias')
                        assert torch.allclose(submodule.weight, hf_sd[weight],
                                              atol=1e-8)
                        assert torch.allclose(submodule.bias, hf_sd[bias], atol=1e-8)




@pytest.mark.parametrize("idx", range(len(model.transformer.layers)))
def test_attention(example_input, idx):
    attn = model.transformer.layers[idx].attention
    ref_attn = ref_model.gpt_neox.layers[idx].attention

    def _reference_split_qkv(qkv, ref_attn):
        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (
            ref_attn.num_attention_heads, 3 * ref_attn.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : ref_attn.head_size].permute(0, 2, 1, 3)
        key = qkv[..., ref_attn.head_size: 2 * ref_attn.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * ref_attn.head_size:].permute(0, 2, 1, 3)

        return query, key, value

    def _ref_get_rotary_params(query, key, value, ref_attn):
        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : ref_attn.rotary_ndims]
        query_pass = query[..., ref_attn.rotary_ndims:]
        key_rot = key[..., : ref_attn.rotary_ndims]
        key_pass = key[..., ref_attn.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        cos, sin = ref_attn.rotary_emb(value, seq_len=seq_len)
        return (query_rot, query_pass), (key_rot, key_pass), cos, sin

    # TODO: This is fragile. Need to make these actual functions and use them,
    #  as a refactor breaks these

    def _ref_apply_rotary_emb_and_conc(ref_tup, position_ids):
        query_rot, query_pass = ref_tup[0]
        key_rot, key_pass = ref_tup[1]
        ref_cos, ref_sin = ref_tup[2], ref_tup[3]
        query, key = apply_rotary_pos_emb(query_rot, key_rot, ref_cos, ref_sin,
                                          position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)
        return query, key

    B, T, C = example_input.size()

    qkv = attn._to_qkv(example_input)
    qkv_ref = ref_attn.query_key_value(example_input)

    assert torch.allclose(qkv, qkv_ref,
                          atol=1e-6)

    q, k, v = attn._split_to_qkv_vectors(example_input)
    q_ref, k_ref, v_ref = _reference_split_qkv(qkv_ref, ref_attn)

    assert torch.allclose(q, q_ref,
                          atol=1e-6)

    assert torch.allclose(k, k_ref,
                          atol=1e-6)

    assert torch.allclose(v, v_ref,
                          atol=1e-6)

    tup = attn._get_rotary_params(q, k, v)
    ref_tup = _ref_get_rotary_params(q_ref, k_ref, v_ref, ref_attn)

    q_rot, q_pass = tup[0]
    k_rot, k_pass = tup[1]
    cos, sin = tup[2], tup[3]

    ref_q_rot, ref_q_pass = ref_tup[0]
    ref_k_rot, ref_k_pass = ref_tup[1]
    ref_cos, ref_sin = ref_tup[2], ref_tup[3]

    assert torch.allclose(q_rot, ref_q_rot)
    assert torch.allclose(q_pass, ref_q_pass)
    assert torch.allclose(k_rot, ref_k_rot)
    assert torch.allclose(k_pass, ref_k_pass)
    assert torch.allclose(cos, ref_cos)
    assert torch.allclose(sin, ref_sin)


    q_params = q_rot, q_pass
    k_params = k_rot, k_pass

    position_ids = torch.arange(0, example_input.size()[1]).unsqueeze(0)

    q, k = attn._apply_rotary_embeddings_and_concatenate(q_params, k_params, cos, sin)
    query, key = _ref_apply_rotary_emb_and_conc(ref_tup, position_ids)


    assert torch.allclose(q, query)
    assert torch.allclose(k, key)

    import torch.nn.functional as F

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref_y = F.scaled_dot_product_attention(query, key, v_ref, is_causal=True)

    assert torch.allclose(y, ref_y)

    y = y.transpose(1, 2).contiguous().view(B, T, C)
    output = attn.dense(y)

    ref_y = ref_y.transpose(1, 2).contiguous()
    ref_y = ref_y.view(B, T, C)
    ref_output = ref_attn.dense(ref_y)

    assert torch.allclose(output, ref_output, atol=1e-8)

    ## Finally, test the pass itself
    full_output = attn.forward(example_input)
    full_ref_output = ref_attn.forward(example_input, attention_mask=None,
                                       position_ids=position_ids)

    assert torch.allclose(full_output, full_ref_output[0], atol=1e-8)


def test_layer(example_input):
    hidden_state_ref = example_input
    hidden_state = example_input

    position_ids = torch.arange(0, example_input.size()[1]).unsqueeze(0)

    assert torch.allclose(hidden_state, hidden_state_ref, atol=1e-8)

    for i, (layer, layer_ref) in enumerate(zip(model.transformer.layers, ref_model.gpt_neox.layers)):
        hidden_state_ref = layer_ref(hidden_state_ref, position_ids=position_ids)[0]
        hidden_state = layer(hidden_state)
        assert torch.allclose(hidden_state, hidden_state_ref, atol=1e-8)


@pytest.mark.parametrize("idx", range(len(model.transformer.layers)))
def test_forward(example_input, idx):
    attn = model.transformer.layers[idx].attention
    ref_attn = ref_model.gpt_neox.layers[idx].attention

    position_ids = torch.arange(0, example_input.size()[1]).unsqueeze(0)

    x0 = example_input

    x_1 = model.transformer.layers[idx].input_layernorm(x0)
    x_2 = ref_model.gpt_neox.layers[idx].input_layernorm(x0)


    assert torch.allclose(x_1, x_2, atol=1e-8)

    x_1 = attn.forward(x_1)
    x_2 = ref_attn.forward(x_2, attention_mask=None,
                                       position_ids=position_ids)[0]


    x_1 = x0 + x_1
    x_2 = x0 + x_2

    assert torch.allclose(x_1, x_2, atol=1e-8)

    x_1 = x_1 + model.transformer.layers[idx].mlp(
        model.transformer.layers[idx].post_attention_layernorm(x_1))
    x_2 = x_2 + ref_model.gpt_neox.layers[idx].mlp(
        ref_model.gpt_neox.layers[idx].post_attention_layernorm(x_2))

    assert torch.allclose(x_1, x_2, atol=1e-8)

    x_2 = ref_model.gpt_neox.final_layer_norm(x_2)
    x_1 = model.transformer.final_layer_norm(x_1)

    assert torch.allclose(x_1, x_2, atol=1e-8)


    # Finally, test a formal forward pass


    x_0 = torch.randint(0, 100, (1, 10))
    logits = model.forward(x_0)
    ref_logits = ref_model.forward(x_0, inputs_embeds=None).logits


    assert torch.allclose(logits, ref_logits, atol=1e-8)







def test_qkv_projection_to_rope():
    assert True


def test_roped_to_attention():
    assert True

def test_output_hf(example_input):
    example_input = example_input.long()  # Convert to LongTensor
    ref_model.forward(input_ids=example_input.squeeze(0))