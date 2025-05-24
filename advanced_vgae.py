import torch
from torch_geometric.data import Batch
from torch_geometric.nn import VGAE, GATConv
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

# ----------- Define Enhanced VGAE Model -----------
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=2, concat=True, dropout=0.2)
        self.conv_mu = GATConv(2 * out_channels * 2, out_channels, heads=1, concat=False, dropout=0.2)
        self.conv_logstd = GATConv(2 * out_channels * 2, out_channels, heads=1, concat=False, dropout=0.2)

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
model = VGAE(GATEncoder(in_channels, 32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# ----------- Training Loop -----------
best_val_loss = float('inf')
patience, patience_counter = 20, 0

for epoch in range(1, 501):
    loss = train(model, optimizer, train_data)
    val_loss = evaluate(model, val_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "Data/Output/best_vgae_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(" Early stopping triggered.")
            break

# ----------- Save Final Embeddings -----------
model.load_state_dict(torch.load("Data/Output/best_vgae_model.pth"))
model.eval()
with torch.no_grad():
    z = model.encode(train_data.x, train_data.edge_index)

torch.save(z, "Data/Output/latent_node_embeddings.pt")
print("Saved final VGAE embeddings and best model weights.")
