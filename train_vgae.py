import torch
from torch_geometric.data import DataLoader, Batch
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
import os

# Load processed data
graph_data = torch.load("X:/Backup/RMIT/Sem_4/Project/Code_base/processed_apartment_graphs.pt", weights_only=False)
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

# Filter graphs with valid edges
graph_data = [
    g for g in graph_data
    if hasattr(g, 'edge_index') and isinstance(g.edge_index, torch.Tensor)
    and g.edge_index.dim() == 2 and g.edge_index.size(1) > 0
]
print(f"Filtered to {len(graph_data)} graphs with valid edges.")

# Merge all graphs into one batch
full_batch = Batch.from_data_list(graph_data)

# Save the corresponding layout_area_type labels
if hasattr(full_batch, "layout_area_type"):
    layout_labels = full_batch.layout_area_type.cpu().numpy()

    # Load label classes
    label_path = "X:/Backup/RMIT/Sem_4/Project/Code_base/label_classes.txt"
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        decoded_labels = [classes[i] if i < len(classes) else "unknown" for i in layout_labels]
    else:
        print("⚠️ label_classes.txt not found, saving raw integers")
        decoded_labels = layout_labels

    import pandas as pd
    labels_df = pd.DataFrame({"layout_area_type": decoded_labels})
    labels_df.to_csv("X:/Backup/RMIT/Sem_4/Project/Code_base/node_labels_filtered.csv", index=False)
    print("Saved filtered node labels to 'node_labels_filtered.csv'")
else:
    print("⚠️ layout_area_type attribute not found on the batch.")


# Split edges for VGAE training
transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
train_data, val_data, test_data = transform(full_batch)

# Init model
in_channels = full_batch.num_node_features
model = VGAE(GCNEncoder(in_channels, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
for epoch in range(1, 201):
    loss = train(model, optimizer, train_data)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# Final embeddings
model.eval()
with torch.no_grad():
    z = model.encode(train_data.x, train_data.edge_index)

# Save latent node embeddings
torch.save(z, "X:/Backup/RMIT/Sem_4/Project/Code_base/latent_node_embeddings.pt")
print("Saved latent embeddings to 'latent_node_embeddings.pt'")
