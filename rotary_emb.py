import torch
import numpy as np



def create_rotation_block(angle, dtype):
    return torch.tensor([
        [np.cos(angle), -1 * np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=dtype)


def create_rotation_matrix(embeddings: torch.Tensor, dtype, angle=10000):
    seq_length, embedding_dim = embeddings.shape

    theta = lambda x: np.power(angle, -2 * x / embedding_dim)

    rotation_matrix = torch.zeros((embedding_dim, embedding_dim),
                                  dtype=dtype)
    for pos in range(embedding_dim // 2):
        rotation = theta(pos)
        block = create_rotation_block(rotation, dtype)
        rotation_matrix[2 * pos:2 * pos + 2, 2 * pos:2 * pos + 2] = block

    assert torch.allclose(rotation_matrix[-1][-2:], block[1], atol=1e4)
    return rotation_matrix

class RotaryEmbeddings:

    def __init__(self, config):
        self.config = config

    def __call__(self, *args, **kwargs):
        return create_rotation_matrix(*args, **kwargs,
                                      dtype=self.config.dtype,
                                      angle=self.config.rotary_angle)

