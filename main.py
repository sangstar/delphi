from transformers import AutoModelForCausalLM
import torch
from architecture.config import Config
from architecture.attention import GPTNeoXAttention




def main():
    model_str = "EleutherAI/pythia-14m"
    model = AutoModelForCausalLM.from_pretrained(model_str)
    config = Config(model.state_dict(), "cpu", 128)
    attn = GPTNeoXAttention(config, "gpt_neox.layers.0.attention.query_key_value")
    x = torch.rand(1, 10, config.n_embd)
    batch_size, seq_len, n_embd = x.size()
    attn.forward(x)



if __name__ == "__main__":
    main()