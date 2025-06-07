import torch
from torch_geometric.data import Batch
from torch_geometric.nn import VGAE
from torch_geometric.transforms import RandomLinkSplit
from sklearn.cluster import KMeans
import pandas as pd
import umap.umap_ as umap
from encoder import GATEncoder

# ----------- Load Graph Data -----------
graph_data = torch.load("Data/Output/processed_apartment_graphs.pt", weights_only=False)
# Filter out graphs without edges
graph_data = [g for g in graph_data if hasattr(g, 'edge_index') and g.edge_index.size(1) > 0]
full_batch = Batch.from_data_list(graph_data)

# Save node labels for later visualization if available
if hasattr(full_batch, "layout_area_type"):
    pd.DataFrame({"layout_area_type": full_batch.layout_area_type.cpu().numpy()}).to_csv(
        "Data/Output/node_labels_filtered.csv", index=False)

# ----------- Train/Validation Split -----------
transform = RandomLinkSplit(is_undirected=True, split_labels=True, add_negative_train_samples=True)
train_data, val_data, _ = transform(full_batch)

# ----------- Initialize Model -----------
LATENT_DIM = 64
in_channels = full_batch.num_node_features
model = VGAE(GATEncoder(in_channels, LATENT_DIM))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# ----------- Training Loop with Early Stopping -----------
best_val_loss = float('inf')
patience, patience_counter = 30, 0
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.edge_index)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        z_val = model.encode(val_data.x, val_data.edge_index)
        val_loss = model.recon_loss(z_val, val_data.edge_index).item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:04d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "Data/Output/best_cluster_vgae.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

# ----------- Save Final Embeddings -----------
model.load_state_dict(torch.load("Data/Output/best_cluster_vgae.pth"))
model.eval()
with torch.no_grad():
    z = model.encode(full_batch.x, full_batch.edge_index)

torch.save(z, "Data/Output/latent_node_embeddings_cluster.pt")

# ----------- Perform KMeans Clustering on Embeddings -----------
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(z.cpu().numpy())

pd.DataFrame({"cluster": cluster_labels}).to_csv("Data/Output/embedding_clusters.csv", index=False)

# ----------- Optional: Save 2D UMAP Projection -----------
z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z.cpu().numpy())

pd.DataFrame(z_umap, columns=["UMAP1", "UMAP2"]).to_csv("Data/Output/umap_embeddings.csv", index=False)