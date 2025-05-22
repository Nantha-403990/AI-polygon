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

# ----------- Load Labels -----------
labels = pd.read_csv("Data/Output/node_labels_filtered.csv")["layout_area_type"].values

# Map numeric labels to room names if available
label_path = "Data/Output/label_classes.txt"
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    label_names = [classes[int(l)] if int(l) < len(classes) else "unknown" for l in labels]
else:
    label_names = labels  # fallback

# Encode room type names to numeric color codes
le = LabelEncoder()
label_codes = le.fit_transform(label_names)

# ----------- Apply UMAP -----------
z_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z_np)

# ----------- Plot UMAP Projection -----------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=label_codes, cmap='tab10', s=10)
plt.legend(handles=scatter.legend_elements()[0], labels=le.classes_.tolist(), title="Room Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("VGAE Latent Embeddings Colored by Room Type")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
