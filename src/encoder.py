import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = self.act(h)
        h = self.conv2(h, edge_index, edge_type)
        return h