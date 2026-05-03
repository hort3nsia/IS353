import pytest
import torch
from torch_geometric.data import Data as PyGData
from src.data_processing import load_biosnap_data, create_data_object

def test_load_biosnap_data():
    data = load_biosnap_data("data/raw")
    assert data.edge_index.shape[0] == 2
    assert data.edge_type.shape[0] > 0
    assert data.num_nodes > 0

def test_create_data_object():
    triplets = [("drug_a", "inhibit", "protein_b")]
    node_to_id = {"drug_a": 0, "protein_b": 1}
    rel_to_id = {"inhibit": 0}
    data = create_data_object(triplets, node_to_id, rel_to_id)
    assert isinstance(data, PyGData)
    assert data.edge_index.shape == (2, 1)
