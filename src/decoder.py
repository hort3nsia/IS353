import torch
import torch.nn as nn

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, hidden_dim):
        super().__init__()
        self.W_r = nn.Parameter(torch.randn(num_relations, hidden_dim))
        
    def forward(self, z_u, z_v, r):
        W_r = self.W_r[r]
        scores = torch.sum(z_u * W_r * z_v, dim=-1)
        return torch.sigmoid(scores)