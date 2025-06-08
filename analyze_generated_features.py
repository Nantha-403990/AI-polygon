import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from graph_decoder import generate_layout_graph

# Settings
LATENT_DIM = 32
NODE_FEATURE_DIM = 33
SAMPLE_SIZE = 50

# Generate a graph
graph = generate_layout_graph(z_dim=LATENT_DIM, out_node_dim=NODE_FEATURE_DIM, total_nodes=SAMPLE_SIZE)

# Save to CSV
df = pd.DataFrame(graph.x.detach().numpy())
df.to_csv("Data/Output/generated_node_features.csv", index=False)
print("✅ Saved generated node features to CSV.")

# Visualize as heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df, cmap="viridis", annot=False)
plt.title("Generated Node Features Heatmap")
plt.xlabel("Feature Index")
plt.ylabel("Node Index")
plt.tight_layout()
plt.show()
