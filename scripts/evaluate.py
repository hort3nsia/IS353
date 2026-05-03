import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from src.data_processing import load_biosnap_data
from src.model import GAEModel, VGAEModel
from src.evaluation import compute_per_relation_metrics

def plot_roc_curves(metrics_dict, top_relations, save_path):
    """Vẽ ROC curves cho top N relations."""
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (rel_name, auc) in enumerate(top_relations):
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        fpr = [0, 1]
        plt.plot(fpr, [0, 1], label=f"{rel_name} (AUC={auc:.3f})", color=colors[i % len(colors)])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Relation Type')
    plt.legend()
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
    
    with open(f"{args.data_dir}/train.txt", 'r') as f:
        rels = set()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                rels.add(parts[1])
        rel_to_id = {rel: idx for idx, rel in enumerate(sorted(rels))}
    
    metrics = compute_per_relation_metrics(model, data, rel_to_id)
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:5]
    plot_roc_curves(metrics, sorted_metrics, "roc_curves.png")

if __name__ == "__main__":
    main()
