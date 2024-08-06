import torch.nn as nn
import architecture as arch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
import torch

class GPTNeoXModel:

    def __init__(self, config, tokenizer=None):
        self.config = config
        self.num_transformer_layers = self.config.num_hidden_layers
        self.tokenizer = tokenizer

        # Load model weights in to arch modules
        self.transformer = nn.ModuleDict(dict(
            emb_in=arch.Embedding(config, "gpt_neox.embed_in"),
            h=nn.ModuleList((arch.TransformerBlock(config, i)) for i in
                            range(self.num_transformer_layers)),
            ln_f=arch.LayerNorm(config, "gpt_neox.final_layer_norm"),
            emb_out=arch.Embedding(config, "embed_out")
        ))

        # Pythia doesn't have a lm_head, use weights from emb_out for lm_head
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.emb_out.weight

    # TODO: Allow this to be loaded locally/over HTTP with tensorizer, use
    #       no_init_or_tensor to avoid double-loading
    @classmethod
    def from_pretrained(cls, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = Config(model.state_dict(), model.config, dtype=model.dtype)
        config.state_dict = model.state_dict()
        return cls(config, tokenizer)

    def forward(self, x):
        x = self.transformer.emb_in(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, prompt, max_length=50, temperature=2, repetition_penalty=10, top_k=1000):
        assert self.tokenizer, "Tokenizer is needed for generation"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_ids = input_ids
        remembered_ids = []

        for _ in range(max_length):
            logits = self.forward(generated_ids)
            logits = logits[:, -1, :] / temperature


            if len(remembered_ids) > 0:
                for token_id in remembered_ids:
                    logits[:, token_id] /= repetition_penalty

            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities, ids = probabilities.sort(dim=-1, descending=True)
            probabilities, ids = probabilities[:, :top_k], ids[:, :top_k]
            next_token_id = ids[:, torch.multinomial(probabilities, num_samples=1)]
            remembered_ids.append(next_token_id)
            generated_ids = torch.cat([generated_ids, next_token_id[:,:,-1]], dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids[0],
                                               skip_special_tokens=True)
        return generated_text

