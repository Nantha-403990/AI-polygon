import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # You missed this import
import os

# Load list of embeddings
z_list = torch.load("latent_node_embeddings.pt", weights_only=False)

# Stack into a single tensor
if isinstance(z_list, list):
    z = torch.cat(z_list, dim=0)
else:
    z = z_list

z_np = z.cpu().numpy()
print(f"Loaded {z_np.shape[0]} node embeddings of dimension {z_np.shape[1]}")

# UMAP to 2D
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
z_2d = umap_model.fit_transform(z_np)

# Plot raw UMAP
plt.figure(figsize=(10, 7))
plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.6, cmap='viridis')
plt.title("VGAE Latent Node Embeddings (UMAP Projection)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Check if label file exists
if os.path.exists("node_labels.csv"):
    labels = pd.read_csv("node_labels.csv")["layout_area_type"]

    # UMAP again (optional if z_2d already exists)
    z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(z_np)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=labels.astype('category').cat.codes, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Room Type")
    plt.title("Latent Space Colored by Room Type")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Warning: 'node_labels.csv' not found. Skipping colored plot.")
