import pytest
import torch
from src.decoder import DistMultDecoder

def test_distmult_decoder():
    num_relations = 10
    hidden_dim = 64
    decoder = DistMultDecoder(num_relations, hidden_dim)
    z_u = torch.randn(32, hidden_dim)
    z_v = torch.randn(32, hidden_dim)
    r = torch.randint(0, num_relations, (32,))
    scores = decoder(z_u, z_v, r)
    assert scores.shape == (32,)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)