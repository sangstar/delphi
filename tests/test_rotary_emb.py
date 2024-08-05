import numpy as np
import torch
from transformers import AutoModelForCausalLM

from rotary_emb import create_rotation_matrix

model_str = "EleutherAI/pythia-14m"
model = AutoModelForCausalLM.from_pretrained(model_str)
data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

embeddings = torch.tensor(data)


def test_rotation_matrix_construction():
    expected_matrix = torch.tensor(
        [
            [0.5403, -0.8415, 0, 0],
            [0.8415, 0.5403, 0, 0],
            [0, 0, 0.99995, -0.01],
            [0, 0, 0.01, 0.99995]
        ])
    r_matrix = create_rotation_matrix(embeddings)
    assert torch.allclose(r_matrix, expected_matrix, atol=1e-4)



