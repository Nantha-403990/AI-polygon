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
OUTPUT_DIR = "Data/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------- FEATURES -----------
NUMERIC_FEATURES = [
    'building_footprint_area', 'number_of_floors', 'floor_area', 'number_of_units',
    'floor_number', 'layout_area', 'layout_biggest_rectangle_length',
    'layout_biggest_rectangle_width', 'layout_compactness', 'layout_door_perimeter',
    'layout_mean_walllengths', 'layout_net_area', 'layout_number_of_doors',
    'layout_number_of_windows', 'layout_open_perimeter', 'layout_perimeter',
    'layout_railing_perimeter', 'layout_room_count', 'layout_std_walllengths',
    'layout_window_perimeter'
]

BINARY_FEATURES = [
    'layout_connects_to_bathroom', 'layout_connects_to_private_outdoor',
    'layout_has_bathtub', 'layout_has_entrance_door', 'layout_has_shower',
    'layout_has_sink', 'layout_has_stairs', 'layout_has_toilet', 'layout_is_navigable'
]

CATEGORICAL_FEATURE = 'entity_subtype'  # to be encoded


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
        for node in G.nodes:
            G.nodes[node]['degree'] = G.degree[node]  # topological feature
        graphs[apt_id] = G
    return graphs


def graph_to_pyg(graph, layout_le, subtype_le):
    node_ids = list(graph.nodes())
    node_attrs = [graph.nodes[n] for n in node_ids]

    def safe_get(k, default=0.0):
        return [n.get(k, default) for n in node_attrs]

    feature_tensors = [torch.tensor(safe_get(f), dtype=torch.float) for f in NUMERIC_FEATURES]
    binary_tensors = [torch.tensor(safe_get(f), dtype=torch.float) for f in BINARY_FEATURES]
    subtype_vals = safe_get(CATEGORICAL_FEATURE, default='unknown')
    subtype_encoded = torch.tensor(subtype_le.transform(subtype_vals), dtype=torch.float)
    degree_tensor = torch.tensor(safe_get('degree'), dtype=torch.float)

    feature_tensors.extend(binary_tensors)
    feature_tensors.append(subtype_encoded)
    feature_tensors.append(degree_tensor)
    feature_tensors.append(torch.tensor(safe_get('centroid_x'), dtype=torch.float))
    feature_tensors.append(torch.tensor(safe_get('centroid_y'), dtype=torch.float))

    x = torch.stack(feature_tensors, dim=1)

    layout_vals = safe_get('layout_area_type', default='unknown')
    y = torch.tensor(layout_le.transform(layout_vals), dtype=torch.long)

    with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "a", newline='') as f:
        writer = csv.writer(f)
        for label in layout_vals:
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


def process_all_partitions():
    all_data = []
    all_layout_labels = []
    all_subtype_labels = []
    raw_graphs = []

    with open(os.path.join(OUTPUT_DIR, "node_labels_filtered.csv"), "w", newline='') as f:
        csv.writer(f).writerow(['layout_area_type'])

    for filename in PARTITION_FILES:
        print(f"Processing {filename}")
        gdf = load_partition(os.path.join(PARTITION_DIR, filename))
        graphs = build_apartment_graphs(gdf)
        for graph in graphs.values():
            node_attrs = [n[1] for n in graph.nodes(data=True)]
            all_layout_labels.extend([n.get('layout_area_type', 'unknown') for n in node_attrs])
            all_subtype_labels.extend([n.get(CATEGORICAL_FEATURE, 'unknown') for n in node_attrs])
            raw_graphs.append(graph)

    layout_le = LabelEncoder()
    layout_le.fit(all_layout_labels)
    subtype_le = LabelEncoder()
    subtype_le.fit(all_subtype_labels)

    with open(os.path.join(OUTPUT_DIR, "label_classes.txt"), "w") as f:
        for cls in layout_le.classes_:
            f.write(f"{cls}\n")

    print(f"Found {len(layout_le.classes_)} layout_area_type classes")
    print(f"Found {len(subtype_le.classes_)} entity_subtype classes")

    for graph in raw_graphs:
        try:
            pyg = graph_to_pyg(graph, layout_le, subtype_le)
            if pyg:
                all_data.append(pyg)
        except Exception as e:
            print(f"⚠️ Skipped graph due to error: {e}")

    print(f"✅ Final processed graphs: {len(all_data)}")
    return all_data


if __name__ == "__main__":
    torch.save(process_all_partitions(), os.path.join(OUTPUT_DIR, "processed_apartment_graphs.pt"))
    print("Saved: processed_apartment_graphs.pt")
