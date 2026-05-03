import pytest
import torch
from src.encoder import RGCNEncoder

def test_rgcn_encoder():
    num_nodes = 100
    num_relations = 10
    hidden_dim = 64
    encoder = RGCNEncoder(num_nodes, num_relations, hidden_dim)
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    edge_type = torch.randint(0, num_relations, (500,))
    z = encoder(x, edge_index, edge_type)
    assert z.shape == (num_nodes, hidden_dim)