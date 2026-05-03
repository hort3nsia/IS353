import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def negative_sampling(edge_index, edge_type, num_nodes, num_negatives=5):
    """Sinh negative edges cho mỗi positive edge."""
    pos_u, pos_v = edge_index[0], edge_index[1]
    num_edges = pos_u.shape[0]

    neg_u = pos_u.repeat(num_negatives)
    neg_v = torch.randint(0, num_nodes, (num_edges * num_negatives,))
    neg_r = edge_type.repeat(num_negatives)

    return neg_u, neg_v, neg_r

def evaluate_link_prediction(model, data, num_negatives=5):
    """Đánh giá link prediction với AUC, AUPRC."""
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_type)

        pos_u, pos_v = data.edge_index[0], data.edge_index[1]
        pos_r = data.edge_type
        pos_scores = model.decode(z[pos_u], z[pos_v], pos_r).cpu().numpy()

        neg_u, neg_v, neg_r = negative_sampling(data.edge_index, data.edge_type, data.num_nodes, num_negatives)
        neg_scores = model.decode(z[neg_u], z[neg_v], neg_r).cpu().numpy()

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])

        auc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

    return {"auc": auc, "auprc": auprc}

def compute_per_relation_metrics(model, data, rel_to_id):
    """Tính AUC riêng cho từng relation type."""
    model.eval()
    results = {}
    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_type)
        for rel_name, rel_id in rel_to_id.items():
            mask = data.edge_type == rel_id
            if mask.sum() < 10:
                continue
            pos_u = data.edge_index[0][mask]
            pos_v = data.edge_index[1][mask]
            pos_scores = model.decode(z[pos_u], z[pos_v], data.edge_type[mask]).cpu().numpy()

            neg_u, neg_v, neg_r = negative_sampling(
                data.edge_index[:, mask],
                data.edge_type[mask],
                data.num_nodes
            )
            neg_scores = model.decode(z[neg_u], z[neg_v], neg_r).cpu().numpy()

            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])

            if len(np.unique(y_true)) > 1:
                results[rel_name] = roc_auc_score(y_true, y_scores)
    return results