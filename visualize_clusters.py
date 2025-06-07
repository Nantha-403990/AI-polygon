import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load UMAP projection and clustering labels
umap_df = pd.read_csv("Data/Output/umap_embeddings.csv")
cluster_labels = pd.read_csv("Data/Output/embedding_clusters.csv")["cluster"]

# Combine for easier plotting
dumap_clustered = umap_df.copy()
dumap_clustered["cluster"] = cluster_labels

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=dumap_clustered, x="UMAP1", y="UMAP2", hue="cluster", palette="tab10", s=20)
plt.title("UMAP Projection of Node Embeddings with KMeans Clusters")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()


# Load data
labels_df = pd.read_csv("Data/Output/node_labels_filtered.csv")
clusters_df = pd.read_csv("Data/Output/embedding_clusters.csv")

# Combine into a single DataFrame
merged_df = labels_df.copy()
merged_df["cluster"] = clusters_df["cluster"]

# Count occurrences of each label per cluster
cluster_summary = merged_df.groupby("cluster")["layout_area_type"].value_counts().unstack(fill_value=0)

# Save the summary to CSV
cluster_summary.to_csv("Data/Output/cluster_label_distribution.csv")
print("✅ Saved cluster-label breakdown to cluster_label_distribution.csv")

# Visualize as heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Room Type Distribution per Cluster")
plt.ylabel("Cluster")
plt.xlabel("Layout Area Type")
plt.tight_layout()
plt.show()