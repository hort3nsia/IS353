import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def negative_sampling(edge_index, edge_type, num_nodes, num_negatives=5):
    """Sinh negative edges cho mỗi positive edge."""
    pos_u, pos_v = edge_index[0], edge_index[1]
    num_edges = pos_u.shape[0]

    neg_u = pos_u.repeat(num_negatives)
    neg_v = torch.randint(0, num_nodes, (num_edges * num_negatives,))
    neg_r = edge_type.repeat(num_negatives)

    return neg_u, neg_v, neg_r

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pos_scores, neg_scores):
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        return self.bce(all_scores, all_labels)

def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index, data.edge_type)

    pos_u = data.edge_index[0]
    pos_v = data.edge_index[1]
    pos_r = data.edge_type

    pos_scores = model.decode(z[pos_u], z[pos_v], pos_r)
    neg_u, neg_v, neg_r = negative_sampling(data.edge_index, data.edge_type, data.num_nodes)
    neg_scores = model.decode(z[neg_u], z[neg_v], neg_r)

    loss = criterion(pos_scores, neg_scores)
    loss.backward()
    optimizer.step()
    return loss.item()