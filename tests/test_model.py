import pytest
import torch
from src.model import GAEModel, VGAEModel

def test_gae_model():
    num_nodes, num_relations, hidden_dim = 100, 10, 64
    model = GAEModel(num_nodes, num_relations, hidden_dim)
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    edge_type = torch.randint(0, num_relations, (500,))
    z = model(x, edge_index, edge_type)
    assert z.shape == (num_nodes, hidden_dim)

def test_vgae_model():
    num_nodes, num_relations, hidden_dim = 100, 10, 64
    model = VGAEModel(num_nodes, num_relations, hidden_dim)
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, 500))
    edge_type = torch.randint(0, num_relations, (500,))
    z_mean, z_logvar = model.encode(x, edge_index, edge_type)
    assert z_mean.shape == z_logvar.shape == (num_nodes, hidden_dim)
