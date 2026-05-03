import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.data_processing import load_biosnap_data
from src.model import GAEModel, VGAEModel

def plot_tsne(z, node_types, save_path):
    """t-SNE visualization of node embeddings."""
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(12, 10))
    unique_types = list(set(node_types))
    colors = ['red' if t == 'protein' else 'blue' for t in unique_types]

    for t, color in zip(unique_types, colors):
        mask = [nt == t for nt in node_types]
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1],
                    c=color, alpha=0.6, label=t.capitalize(), s=50)
    plt.legend()
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="gae", choices=["gae", "vgae"])
    args = parser.parse_args()

    data = load_biosnap_data(args.data_dir)
    num_relations = data.edge_type.max().item() + 1

    if args.model_type == "gae":
        model = GAEModel(data.num_nodes, num_relations, 64)
    else:
        model = VGAEModel(data.num_nodes, num_relations, 64)

    model.load_state_dict(torch.load(args.model_path))

    x = torch.eye(data.num_nodes)
    with torch.no_grad():
        if args.model_type == "vgae":
            z, _, _ = model(x, data.edge_index, data.edge_type)
        else:
            z = model(x, data.edge_index, data.edge_type)

    node_types = {i: 'drug' if i % 2 == 0 else 'protein' for i in range(data.num_nodes)}
    plot_tsne(z.cpu().numpy(), node_types, "tsne_visualization.png")

if __name__ == "__main__":
    main()