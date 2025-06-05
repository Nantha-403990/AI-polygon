## Step-by-Step Instructions

### 1.  Preprocess Graph Data
If you have raw CSV files:
```bash
python Process_csv.py
```
- Converts CSVs to PyTorch Geometric graphs (`processed_apartment_graphs.pt`)

### 2. Train the VGAE Model
```bash
python train_vgae.py
```
- Trains a VGAE model using the GATEncoder
- Saves latent embeddings and UMAP projection

### 3.  Generate & Visualize Layouts
```bash
python generate_and_plot.py
```
- Uses the trained decoder to sample new layout graphs
- Visualizes the graph structure
- Prints node features for inspection

---

##  Requirements
Install dependencies using pip:
```bash
pip install torch torch_geometric pandas matplotlib umap-learn networkx scikit-learn shapely geopandas
```

---

##  Notes
- `encoder.py`, `graph_decoder.py`, and `visualize_graph.py` must be in the same directory as the main scripts.
- You can adjust latent dimensions and node feature dimensions inside `generate_and_plot.py` to match your model.