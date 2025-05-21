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

PARTITION_DIR = "X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/partitioned_merged_apartments/new_partition"  
PARTITION_FILES = [f for f in os.listdir(PARTITION_DIR) if f.endswith(".csv")]
FEATURES = ['building_footprint_area', 'number_of_floors', 'floor_area', 'number_of_units']
LABEL_COLUMN = 'dwelling_type'

OUTPUT_DIR = "Data/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- HELPER FUNCTIONS -----------
# ----------- HELPER FUNCTIONS -----------

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
        print(f"working {apt_id} ")

        graphs[apt_id] = G
    return graphs

def graph_to_pyg(graph):
    node_ids = list(graph.nodes())
    node_attrs = [graph.nodes[n] for n in node_ids]

    def safe_get(k, default=0.0): return [n.get(k, default) for n in node_attrs]


    feature_tensors = [torch.tensor(safe_get(f), dtype=torch.float) for f in FEATURES]
    feature_tensors += [
        torch.tensor(safe_get('centroid_x'), dtype=torch.float),
        torch.tensor(safe_get('centroid_y'), dtype=torch.float)
    ]


    x = torch.stack(feature_tensors, dim=1)

    # Encode dwelling_type
    types = [n.get(LABEL_COLUMN, 'unknown') for n in node_attrs]
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(types), dtype=torch.long)

    global label_writer_initialized
    if not label_writer_initialized:
        with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([LABEL_COLUMN])
        label_writer_initialized = True

    with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "a", newline='') as f:
        writer = csv.writer(f)
        for label in types:
            writer.writerow([label])

    with open(os.path.join(OUTPUT_DIR, "label_classes.txt"), "w") as f:
        for c in le.classes_:
            f.write(c + "\n")

    edge_index = []
    for src, dst in graph.edges():
        src_idx = node_ids.index(src)
        dst_idx = node_ids.index(dst)
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])

    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Write raw labels for external inspection
    global label_writer_initialized
    if not globals().get('label_writer_initialized', False):
        with open("X:/Backup/RMIT/Sem_4/Project/Code_base/node_labels.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layout_area_type'])  # header
        label_writer_initialized = True

    with open("X:/Backup/RMIT/Sem_4/Project/Code_base/node_labels.csv", "a", newline='') as f:
        writer = csv.writer(f)
        for label in types:
            writer.writerow([label])

    # ✅ ATTACH label tensor to the Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.dwelling_type = y
    return data

# ----------- MAIN LOOP -----------

label_writer_initialized = False

# ----------- MAIN LOOP -----------

label_writer_initialized = False

def process_all_partitions():
    all_data = []
    for filename in PARTITION_FILES:
        print(f"Processing {filename}")
        file_path = os.path.join(PARTITION_DIR, filename)
        gdf = load_partition(file_path)
        file_path = os.path.join(PARTITION_DIR, filename)
        gdf = load_partition(file_path)
        graphs = build_apartment_graphs(gdf)

        for apt_id, graph in graphs.items():
            try:
                pyg_graph = graph_to_pyg(graph)
                if pyg_graph is not None:
                    all_data.append(pyg_graph)
            except Exception as e:
                print(f"Skipping {apt_id} due to error: {e}")

    print(f"Total processed graphs: {len(all_data)}")
    return all_data

# ----------- ENTRY POINT -----------


if __name__ == "__main__":
    data_list = process_all_partitions()
    torch.save(data_list, os.path.join(OUTPUT_DIR, "processed_apartment_graphs.pt"))
    print("✅ Saved: processed_apartment_graphs.pt")
