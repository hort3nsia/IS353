import argparse
import torch
from src.data_processing import load_biosnap_data
from src.model import GAEModel, VGAEModel
from src.training import train_epoch, BCELoss
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gae", choices=["gae", "vgae"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    data = load_biosnap_data(args.data_dir)

    num_relations = data.edge_type.max().item() + 1
    model = GAEModel(data.num_nodes, num_relations, args.hidden_dim) if args.model == "gae" \
        else VGAEModel(data.num_nodes, num_relations, args.hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCELoss()

    for epoch in tqdm(range(args.epochs)):
        loss = train_epoch(model, data, optimizer, criterion)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    torch.save(model.state_dict(), f"models/{args.model}.pt")

if __name__ == "__main__":
    main()