import os
import pandas as pd
import geopandas as gpd
import torch
import networkx as nx
from shapely import wkt
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import csv

# ----------- CONFIGURATION -----------
PARTITION_DIR = "Data/merged"
PARTITION_FILES = [f for f in os.listdir(PARTITION_DIR) if f.endswith(".csv")]
FEATURES = ['layout_area', 'layout_compactness', 'layout_is_navigable',
            'view_sky_p80', 'sun_201806211200_median', 'noise_traffic_day']
OUTPUT_DIR = "Data/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- GLOBAL INIT -----------
label_writer_initialized = False
global_labels = []  # collect all types before fitting LabelEncoder

# ----------- HELPERS -----------
def load_partition(file_path):
    df = pd.read_csv(file_path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(df['geometry']), crs="EPSG:4326")
    gdf = gdf.dropna(subset=['apartment_id', 'area_id_geom', 'geometry'])
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)
    gdf['centroid_x'] = gdf['geometry'].centroid.x
    gdf['centroid_y'] = gdf['geometry'].centroid.y
    return gdf

def build_apartment_graphs(gdf):
    graphs = {}
    for apt_id, group in gdf.groupby("apartment_id"):
        G = nx.Graph()
        for _, row in group.iterrows():
            G.add_node(row['area_id_geom'], **row.to_dict())
        for i, r1 in group.iterrows():
            for j, r2 in group.iloc[i+1:].iterrows():
                if r1['geometry'].intersects(r2['geometry']):
                    G.add_edge(r1['area_id_geom'], r2['area_id_geom'])
        graphs[apt_id] = G
    return graphs

def graph_to_pyg(graph, le):
    node_ids = list(graph.nodes())
    node_attrs = [graph.nodes[n] for n in node_ids]

    def safe_get(k, default=0.0): return [n.get(k, default) for n in node_attrs]
    feature_tensors = [torch.tensor(safe_get(f), dtype=torch.float) for f in FEATURES]
    feature_tensors += [
        torch.tensor(safe_get('centroid_x'), dtype=torch.float),
        torch.tensor(safe_get('centroid_y'), dtype=torch.float)
    ]
    x = torch.stack(feature_tensors, dim=1)

    raw_types = [n.get('layout_area_type', 'unknown') for n in node_attrs]
    y = torch.tensor(le.transform(raw_types), dtype=torch.long)

    global label_writer_initialized
    if not label_writer_initialized:
        with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "w", newline='') as f:
            csv.writer(f).writerow(['layout_area_type'])
        label_writer_initialized = True

    with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "a", newline='') as f:
        writer = csv.writer(f)
        for label in raw_types:
            writer.writerow([label])

    edge_index = []
    for src, dst in graph.edges():
        src_idx = node_ids.index(src)
        dst_idx = node_ids.index(dst)
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y)
    data.layout_area_type = y
    return data

# ----------- MAIN WORKFLOW -----------
def process_all_partitions():
    all_data, all_types = [], []
    raw_graphs = []

    for filename in PARTITION_FILES:
        print(f"Processing {filename}")
        gdf = load_partition(os.path.join(PARTITION_DIR, filename))
        graphs = build_apartment_graphs(gdf)
        for graph in graphs.values():
            raw_types = [n[1].get('layout_area_type', 'unknown') for n in graph.nodes(data=True)]
            all_types.extend(raw_types)
            raw_graphs.append(graph)

    le = LabelEncoder()
    le.fit(all_types)
    print(f"✅ Found {len(le.classes_)} layout types")

    with open(os.path.join(OUTPUT_DIR, "label_classes.txt"), "w") as f:
        for cls in le.classes_:
            f.write(f"{cls}\n")

    final_graphs = []
    for graph in raw_graphs:
        try:
            pyg = graph_to_pyg(graph, le)
            if pyg:
                final_graphs.append(pyg)
        except Exception as e:
            print(f"⚠️ Skipped graph due to error: {e}")
    print(f"✅ Final processed graphs: {len(final_graphs)}")
    return final_graphs

# ----------- ENTRY POINT -----------
if __name__ == "__main__":
    torch.save(process_all_partitions(), os.path.join(OUTPUT_DIR, "processed_apartment_graphs.pt"))
    print("✅ Saved: processed_apartment_graphs.pt")
