import torch
from collections import defaultdict

def get_neighborhood(edge_index, edge_type, node_id, relation_to_id, depth=2):
    """Lấy neighbors bậc 1 và bậc 2 của một node."""
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        r = edge_type[i].item()
        adj[u].append((v, r))
        adj[v].append((u, r))

    neighborhoods = {1: [], 2: []}
    visited = {node_id}
    current = {node_id}

    for d in range(1, depth + 1):
        next_nodes = set()
        for n in current:
            for neighbor, rel in adj[n]:
                if neighbor not in visited:
                    neighborhoods[d].append((neighbor, rel))
                    next_nodes.add(neighbor)
                    visited.add(neighbor)
        current = next_nodes

    return neighborhoods

def case_study(model, data, drug_name, protein_name, drug_id, protein_id, relation="inhibit"):
    """Phân tích case study cho một triplet."""
    with torch.no_grad():
        z = model(data.x, data.edge_index, data.edge_type)
        rel_id = list(model.decoder.W_r.shape[0] - 1)
        score = model.decode(z[drug_id], z[protein_id], torch.tensor([rel_id])).item()

    neighborhoods = get_neighborhood(data.edge_index, data.edge_type, drug_id, {})

    return {
        "drug": drug_name,
        "protein": protein_name,
        "prediction_score": score,
        "neighborhood_1hop": neighborhoods[1],
        "neighborhood_2hop": neighborhoods[2]
    }