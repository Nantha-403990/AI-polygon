import os
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ----------- Load Embeddings -----------
z_list = torch.load("Data/Output/latent_node_embeddings.pt", weights_only=False)
z = torch.cat(z_list, dim=0) if isinstance(z_list, list) else z_list
z_np = z.cpu().numpy()
print(f"Loaded {z_np.shape[0]} node embeddings of dimension {z_np.shape[1]}")

# ----------- Load Labels (if available) -----------
label_path = "Data/Output/node_labels_filtered.csv"
if os.path.exists(label_path):
    labels_raw = pd.read_csv(label_path)["layout_area_type"].astype(str)
    le = LabelEncoder()
    label_codes = le.fit_transform(labels_raw)
    label_names = le.classes_
    print(f"Loaded labels: {len(label_names)} classes")
else:
    labels_raw = None
    label_codes = None
    label_names = None
    print("⚠️ Labels not found. Plotting without color.")

# ----------- UMAP Dimensionality Reduction -----------
z_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z_np)

# ----------- Plotting -----------
plt.figure(figsize=(10, 7))
if label_codes is not None:
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=label_codes, cmap='tab10', s=10)
    plt.legend(handles=scatter.legend_elements()[0], labels=label_names, title="Room Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("VGAE Embeddings Colored by Room Type")
else:
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.7)
    plt.title("VGAE Embeddings (Unlabeled)")

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
