import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_biosnap_data(data_dir):
    """Load BioSNAP dataset and convert to Data object."""
    triplets = []
    node_set = set()
    rel_set = set()

    for filename in ["train.txt", "test.txt", "valid.txt"]:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        s, r, o = parts
                        triplets.append((s, r, o))
                        node_set.add(s)
                        node_set.add(o)
                        rel_set.add(r)

    if not triplets:
        return None

    node_to_id = {node: idx for idx, node in enumerate(sorted(node_set))}
    rel_to_id = {rel: idx for idx, rel in enumerate(sorted(rel_set))}

    edges = []
    edge_types = []
    for s, r, o in triplets:
        edges.append([node_to_id[s], node_to_id[o]])
        edge_types.append(rel_to_id[r])

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    num_nodes = len(node_to_id)

    return Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)

def create_data_object(triplets, node_to_id, rel_to_id):
    """Convert triplets list to PyG Data object."""
    edges = []
    edge_types = []
    for s, r, o in triplets:
        edges.append([node_to_id[s], node_to_id[o]])
        edge_types.append(rel_to_id[r])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    num_nodes = len(node_to_id)
    return Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)
