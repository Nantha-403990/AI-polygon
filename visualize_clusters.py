import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load label class names from file
def load_label_classes(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

label_classes = load_label_classes("Data/Output/label_classes.txt")

# Load UMAP projection and clustering labels
umap_df = pd.read_csv("Data/Output/umap_embeddings.csv")
cluster_labels = pd.read_csv("Data/Output/embedding_clusters.csv")["cluster"]

# Combine for UMAP scatter plot
dumap_clustered = umap_df.copy()
dumap_clustered["cluster"] = cluster_labels

# Plot UMAP
plt.figure(figsize=(10, 7))
sns.scatterplot(data=dumap_clustered, x="UMAP1", y="UMAP2", hue="cluster", palette="tab10", s=20)
plt.title("UMAP Projection of Node Embeddings with KMeans Clusters")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

# Load label and cluster assignments
labels_df = pd.read_csv("Data/Output/node_labels_filtered.csv")
clusters_df = pd.read_csv("Data/Output/embedding_clusters.csv")

# Add clusters to the label data
merged_df = labels_df.copy()
merged_df["cluster"] = clusters_df["cluster"]

# Map layout_area_type IDs to actual label names
merged_df["layout_area_type_name"] = merged_df["layout_area_type"].apply(
    lambda x: label_classes[int(x)] if int(x) < len(label_classes) else f"Unknown ({x})"
)

# Count layout label names per cluster
cluster_summary_named = merged_df.groupby("cluster")["layout_area_type_name"].value_counts().unstack(fill_value=0)

# Save as CSV
cluster_summary_named.to_csv("Data/Output/cluster_label_distribution_named.csv")
print("✅ Saved cluster-label breakdown with names to cluster_label_distribution_named.csv")

# Heatmap with names
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary_named, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Room Type Distribution per Cluster (Named Labels)")
plt.ylabel("Cluster")
plt.xlabel("Layout Area Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
