import torch
import torch.nn as nn
from torch_geometric.data import Data

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, out_node_dim):
        super().__init__()
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_node_dim)
        )
        self.edge_decoder = nn.CosineSimilarity(dim=1)

    def forward(self, z):
        x_recon = self.node_decoder(z)
        num_nodes = z.size(0)
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = self.edge_decoder(z[i].unsqueeze(0), z[j].unsqueeze(0))
                if sim > 0.2:  # Lowered threshold
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        # Fallback: if graph is too sparse, connect top-2 similar neighbors
        if not edge_index:
            for i in range(num_nodes):
                sims = [(j, self.edge_decoder(z[i].unsqueeze(0), z[j].unsqueeze(0)).item())
                        for j in range(num_nodes) if i != j]
                top_k = sorted(sims, key=lambda x: x[1], reverse=True)[:2]
                for j, _ in top_k:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        return x_recon, edge_index


def generate_layout_graph(z_dim, out_node_dim, sample_size=10):
    decoder = GraphDecoder(latent_dim=z_dim, out_node_dim=out_node_dim)
    z_sample = torch.randn((sample_size, z_dim))
    x_gen, edge_index_gen = decoder(z_sample)
    return Data(x=x_gen, edge_index=edge_index_gen)
