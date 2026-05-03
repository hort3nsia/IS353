import torch
import torch.nn as nn
from src.encoder import RGCNEncoder
from src.decoder import DistMultDecoder

class GAEModel(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim):
        super().__init__()
        self.encoder = RGCNEncoder(num_nodes, num_relations, hidden_dim)
        self.decoder = DistMultDecoder(num_relations, hidden_dim)
        
    def forward(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)
    
    def decode(self, z_u, z_v, r):
        return self.decoder(z_u, z_v, r)

class VGAEModel(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim):
        super().__init__()
        self.encoder = RGCNEncoder(num_nodes, num_relations, hidden_dim)
        self.decoder = DistMultDecoder(num_relations, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        
    def encode(self, x, edge_index, edge_type):
        h = self.encoder(x, edge_index, edge_type)
        return self.fc_mean(h), self.fc_logvar(h)
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    def forward(self, x, edge_index, edge_type):
        mean, logvar = self.encode(x, edge_index, edge_type)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def decode(self, z_u, z_v, r):
        return self.decoder(z_u, z_v, r)
