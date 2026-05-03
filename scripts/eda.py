import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def plot_relation_distribution(edge_types, rel_to_id, save_path):
    """Vẽ bar chart phân bố số cạnh theo relation type."""
    id_to_rel = {v: k for k, v in rel_to_id.items()}
    rel_counts = {}
    for et in edge_types:
        rel_name = id_to_rel[et.item()]
        rel_counts[rel_name] = rel_counts.get(rel_name, 0) + 1

    df = pd.DataFrame(list(rel_counts.items()), columns=['Relation', 'Count'])
    df = df.sort_values('Count', ascending=False).head(20)

    plt.figure(figsize=(12, 6))
    plt.barh(df['Relation'], df['Count'])
    plt.xlabel('Number of Edges')
    plt.title('Relation Distribution (Top 20)')
    plt.tight_layout()
    plt.savefig(save_path)

def visualize_subgraph(edge_index, edge_type, node_types, num_nodes=30, save_path='subgraph.png'):
    """Visualize subgraph với drug=square, protein=circle."""
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(u, v)

    subgraph = list(G.nodes())[:num_nodes]
    pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))
    drug_nodes = [n for n in subgraph if node_types.get(n) == 'drug']
    protein_nodes = [n for n in subgraph if node_types.get(n) == 'protein']
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_shape='s', node_color='blue')
    nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_shape='o', node_color='red')
    nx.draw_networkx_edges(G, pos)
    plt.savefig(save_path)