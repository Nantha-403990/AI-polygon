import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load list of embeddings
z_list = torch.load("X:/Backup/RMIT/Sem_4/Project/Code_base/latent_node_embeddings.pt", weights_only=False)

# Stack into a single tensor
if isinstance(z_list, list):
    z = torch.cat(z_list, dim=0)
else:
    z = z_list

# Load labels
if os.path.exists("X:/Backup/RMIT/Sem_4/Project/Code_base/node_labels_filtered.csv"):
    labels_df = pd.read_csv("X:/Backup/RMIT/Sem_4/Project/Code_base/node_labels_filtered.csv")
    if "layout_area_type" in labels_df.columns:
        labels = labels_df["layout_area_type"]
    else:
        raise ValueError("Expected 'layout_area_type' column in node_labels.csv")
else:
    print("⚠️ node_labels.csv not found. Proceeding without labels.")
    labels = None

z_np = z.cpu().numpy()
print(f"Loaded {z_np.shape[0]} node embeddings of dimension {z_np.shape[1]}")

# UMAP to 2D
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
z_2d = umap_model.fit_transform(z_np)

# Plot raw UMAP
plt.figure(figsize=(10, 7))
if labels is not None:
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels.astype('category').cat.codes, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Room Type")
    plt.title("Latent Space Colored by Room Type")
else:
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.6)
    plt.title("Latent Node Embeddings (No Labels)")

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
