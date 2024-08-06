import numpy as np
import torch
from rotary_emb import create_rotation_matrix, create_rotation_block
import pytest

data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

embeddings = torch.tensor(data)

dtype = torch.float32

def test_rotation_matrix_construction():
    expected_matrix = torch.tensor(
        [
            [0.5403, -0.8415, 0, 0],
            [0.8415, 0.5403, 0, 0],
            [0, 0, 0.99995, -0.01],
            [0, 0, 0.01, 0.99995]
        ])
    r_matrix = create_rotation_matrix(embeddings, dtype)
    assert torch.allclose(r_matrix, expected_matrix, atol=1e-4)

def test_create_rotation_block():
    angle = np.pi / 4  # 45 degrees
    block = create_rotation_block(angle, dtype)

    expected_block = torch.tensor([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=dtype)

    assert torch.allclose(block, expected_block, atol=1e-6), f"Expected {expected_block}, but got {block}"

def test_create_rotation_matrix():
    embeddings = torch.randn(10, 8)  # Example tensor with seq_length 10 and embedding_dim 8
    rotation_matrix = create_rotation_matrix(embeddings, dtype)

    embedding_dim = embeddings.shape[1]

    # Check the rotation matrix is of the correct shape
    assert rotation_matrix.shape == (embedding_dim, embedding_dim), f"Expected shape {(embedding_dim, embedding_dim)}, but got {rotation_matrix.shape}"

    # Check the first 2x2 block
    pos = 0
    expected_block = create_rotation_block(np.power(10000, -2 * pos / embedding_dim), dtype)
    assert torch.allclose(rotation_matrix[:2, :2], expected_block, atol=1e-6), f"Expected {expected_block}, but got {rotation_matrix[:2, :2]}"

    # Check the last 2x2 block
    pos = (embedding_dim // 2) - 1
    expected_block = create_rotation_block(np.power(10000, -2 * pos / embedding_dim), dtype)
    assert torch.allclose(rotation_matrix[-2:, -2:], expected_block, atol=1e-6), f"Expected {expected_block}, but got {rotation_matrix[-2:, -2:]}"

    # Additional assertion in create_rotation_matrix function
    block = create_rotation_block(np.power(10000, -2 * (embedding_dim // 2 - 1) / embedding_dim), dtype)
    assert torch.allclose(rotation_matrix[-1, -2:], block[1], atol=1e-4), "The final rotation block check failed"

if __name__ == "__main__":
    pytest.main()
