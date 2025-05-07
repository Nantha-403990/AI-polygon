import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
import os
import umap
import matplotlib.pyplot as plt

# Load processed data
graph_data = torch.load("processed_apartment_graphs.pt", weights_only=False)
print(f"Loaded {len(graph_data)} graphs.")

# Define VGAE model
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# VGAE training loop
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

# Prepare a single large batch (merge all graphs)
graph_data = [
    g for g in graph_data
    if hasattr(g, 'edge_index') and isinstance(g.edge_index, torch.Tensor)
    and g.edge_index.dim() == 2 and g.edge_index.size(1) > 0
]
print(f"Filtered to {len(graph_data)} graphs with valid edges.")

from torch_geometric.data import Batch
full_batch = Batch.from_data_list(graph_data)

# Split edges for VGAE training
transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(full_batch)

# Init model
in_channels = full_batch.num_node_features
model = VGAE(GCNEncoder(in_channels, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(1, 201):
    loss = train(model, optimizer, train_data)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# Encode final embeddings
model.eval()
with torch.no_grad():
    z = model.encode(train_data.x, train_data.edge_index)

torch.save(z, "latent_node_embeddings.pt")

# Save embeddings
torch.save(z, "latent_node_embeddings.pt")
print("Saved latent embeddings to 'latent_node_embeddings.pt'")