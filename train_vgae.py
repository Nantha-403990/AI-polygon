import torch
from torch_geometric.data import Batch
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd
import os

# ----------- Load Graph Data -----------

graph_data = torch.load("Data/Output/processed_apartment_graphs.pt", weights_only=False)
print(f"Loaded {len(graph_data)} graphs.")

# Filter graphs with valid edge_index
graph_data = [
    g for g in graph_data
    if hasattr(g, 'edge_index') and isinstance(g.edge_index, torch.Tensor)
    and g.edge_index.dim() == 2 and g.edge_index.size(1) > 0
]
print(f"Filtered to {len(graph_data)} graphs with valid edges.")

# Merge graphs into one big batch
full_batch = Batch.from_data_list(graph_data)

# Save layout_area_type labels for visualization
if hasattr(full_batch, "layout_area_type"):
    layout_labels = full_batch.layout_area_type.cpu().numpy()
    labels_df = pd.DataFrame({"layout_area_type": layout_labels})
    labels_df.to_csv("Data/Output/node_labels_filtered.csv", index=False)
    print(" Saved filtered node labels to 'node_labels_filtered.csv'")
else:
    print(" layout_area_type attribute not found on the batch.")

# ----------- Define VGAE Model -----------

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# ----------- Train and Eval Functions -----------

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        return model.recon_loss(z, data.edge_index).item()

# ----------- Prepare Data -----------

transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(full_batch)

in_channels = full_batch.num_node_features
model = VGAE(GCNEncoder(in_channels, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------- Training Loop -----------

for epoch in range(1, 201):
    loss = train(model, optimizer, train_data)
    if epoch % 20 == 0:
        val_loss = evaluate(model, val_data)
        print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

# ----------- Save Final Embeddings and Model -----------

model.eval()
with torch.no_grad():
    z = model.encode(train_data.x, train_data.edge_index)

torch.save(z, "Data/Output/latent_node_embeddings.pt")
torch.save(model.state_dict(), "Data/Output/vgae_model.pth")
print("Saved VGAE model and latent embeddings.")
