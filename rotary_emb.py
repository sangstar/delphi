import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


# TODO: Technical debt will be incurred if a global `config.dtype` isn't
#       established. Here I manually pass torch.float32

dtype = torch.float32
def create_rotation_block(angle):
    return torch.tensor([
        [np.cos(angle), -1 * np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=dtype)


def create_rotation_matrix(embeddings: torch.Tensor):
    seq_length, embedding_dim = embeddings.shape

    theta = lambda x: np.power(10000, -2 * x / embedding_dim)

    rotation_matrix = torch.zeros((embedding_dim, embedding_dim), dtype=dtype)
    for pos, embedding in zip(range(round(embedding_dim / 2)), embeddings):
        rotation = theta(pos)
        block = create_rotation_block(rotation)
        rotation_matrix[2 * pos, 2 * pos], rotation_matrix[2 * pos, 2 * pos + 1] = \
            block[0]
        rotation_matrix[2 * pos + 1, 2 * pos], rotation_matrix[
            2 * pos + 1, 2 * pos + 1] = block[1]
    assert torch.allclose(rotation_matrix[-1][-2:], block[1], atol=1e4)
    return rotation_matrix
