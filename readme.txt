# Generative AI Building Layout System – Project README

This repository implements a **Graph Neural Network-based Variational Autoencoder (VGAE)** to learn from spatial building layouts and generate new ones. It includes graph preprocessing, training, clustering, and layout synthesis.

---

##  Directory Structure
```
AI-polygon/
├── Data/
│   └── Output/                       # Processed graphs, model outputs, and visualizations
├── Process_csv.py                   # Preprocesses raw CSV into graph data
├── train_vgae.py                    # Trains VGAE model and saves latent embeddings
├── encoder.py                       # GATEncoder class for VGAE
├── graph_decoder.py                 # GraphDecoder and layout generator from latent z
├── generate_and_plot.py            # Generates a layout and visualizes it
├── analyze_generated_features.py   # Heatmap visualization of generated node features
├── clustered_vgae.py               # VGAE + KMeans clustering and UMAP export
├── visualize_clusters.py           # UMAP scatter plot colored by cluster ID
├── analyze_clusters_by_label.py    # Room type distribution per cluster (heatmap)
├── sample_from_cluster.py          # Generate layouts conditioned on cluster
└── README.md                        # This file
```

---

##  Workflow Overview
###  Script Summary Table
| Script | Purpose | Input | Output |
|--------|---------|--------|--------|
| `Preprocessing.py` | Partition & merge raw geometry/simulation data | `geometries.csv`, `simulations.csv` | Partitioned CSVs |
| `format_shapes_typogen_strict.py` | Converts room-type label CSVs to typogen-compatible format | Node labels CSV | `shapes.txt` in Typogen format |
| `visualize_graph.py` | Visualize a PyG graph and inspect features | `Data` object (graph) | Plot + console output |
| `Process_csv.py` | Preprocess CSV into graphs | Partitioned CSVs | Graph `.pt` + node labels |
| `train_vgae.py` | Train baseline VGAE (GCN) | `.pt` graphs | `latent_node_embeddings.pt`, UMAP |
| `advanced_vgae.py`, `clustered_vgae.py` | Train advanced VGAE (GAT + Clustering) | `.pt` graphs | KMeans + UMAP + model weights |
| `encoder.py` | GATEncoder model | Used by advanced training | — |
| `graph_decoder.py` | Decode latent z to graph | Latent vector | Generated graph (x, edge_index) |
| `generate_and_plot.py` | Visualize one decoded graph | Decoder output | Graph plot |
| `analyze_generated_features.py` | Visualize generated features | `x` from graph | Heatmap, `.csv` |
| `visualize_clusters.py`, `visualize_embeddings.py` | Cluster and latent space visualization | UMAP & cluster data | Scatter plots |
| `analyze_clusters_by_label.py` | Cluster → room type breakdown | Cluster + label CSVs | Heatmap + `.csv` |
| `sample_from_cluster.py` | Generate layouts from cluster | KMeans centroid | Generated graph from cluster |


### **Graph Preprocessing** – `Process_csv.py`
- **Input**: Partitioned CSV files of apartment rooms with WKT geometry and metadata
- **Output**:
  - `processed_apartment_graphs.pt`: PyTorch Geometric graph objects
  - `node_labels_filtered.csv`: Room types (`layout_area_type`) per node
  - `label_classes.txt`: Mapping of room type class labels

### **Train Baseline VGAE (GCNConv-based)** – `train_vgae.py`
- **Input**: `.pt` graph data from step 1
- Defines its own internal GCNEncoder using `GCNConv`
- **Output**:
  - Trained VGAE model weights
  - `latent_node_embeddings.pt`: Final node embeddings from the latent space
  - `umap_embeddings.csv`: 2D projection of embeddings for visualization`.pt` graph data from step 1
- **Output**:
  - Trained VGAE model weights
  - `latent_node_embeddings.pt`: Final node embeddings from the latent space
  - `umap_embeddings.csv`: 2D projection of embeddings for visualization

### **Encode Model** – `encoder.py`
- Contains the `GATEncoder` class (multi-layer Graph Attention Network)
- Used inside `clustered_vgae.py` and `advanced_vgae.py`

### **Graph Generation** – `graph_decoder.py`
- Contains `GraphDecoder` and `generate_layout_graph()`
- **Input**: Latent vector `z`
- **Output**: A generated graph with synthetic node features and edge index

### **Visualize a Sampled Graph** – `generate_and_plot.py`
- Calls the decoder to sample and generate a layout
- Plots it using `networkx`
- **Output**: Visual graph window

### **Analyze Generated Features** – `analyze_generated_features.py`
- Plots a heatmap of generated node features for interpretation
- **Input**: `generated_graph.x`
- **Output**: Heatmap + `generated_node_features.csv`

### **Train Advanced VGAE (GAT + Clustering)** – `advanced_vgae.py`, `clustered_vgae.py`
- Runs VGAE training + KMeans clustering on embeddings
- Saves:
  - `embedding_clusters.csv`
  - `umap_embeddings.csv`
- Also supports increased latent space (e.g., 64 dims)

### **Visualize Clusters and Cluster Composition** – `visualize_clusters.py`, `visualize_embeddings.py`
- Loads `umap_embeddings.csv` + cluster labels
- **Output**: UMAP scatter plot with colors by cluster ID

### **Analyze Room Types by Cluster** – `analyze_clusters_by_label.py`
- Cross-tabulates cluster ID vs. room type (`layout_area_type`)
- **Output**:
  - `cluster_label_distribution.csv`
  - Heatmap showing which clusters correspond to which room types

### **Sample Layout from Cluster** – `sample_from_cluster.py`
- Picks a cluster centroid from KMeans
- Generates a layout graph near that centroid
- Visualizes it

---

##  Dependencies
```bash
pip install torch torch_geometric pandas matplotlib seaborn umap-learn networkx scikit-learn shapely geopandas
```

---

##  Notes
- `Preprocessing.py` is used to create partitioned input CSVs from large building datasets.
- `format_shapes_typogen_strict.py` reformats room label sequences for use with Typogen layout language.
- `visualize_graph.py` is a utility used in generation scripts for quick graph visualization and debugging.
- Cluster IDs are unsupervised — you can interpret them using `analyze_clusters_by_label.py`
- Graph generation is latent-driven — use `sample_from_cluster.py` to generate layouts of different styles
- Node features include centroid coordinates, geometric attributes, and binary flags (e.g., has_toilet)

---

##  Future Enhancements
- Add IFC export for 3D BIM integration
- Integrate geometric layout reconstruction
- Use cluster labels as pseudo-labels for supervised learning

---

## 👤 Maintainers
- Nanthakumarr s, Najmunsaquib Shaikh
- Collaborators: RMIT x Earlybuild

For support or collaboration, please raise an issue or contact the project owner.
