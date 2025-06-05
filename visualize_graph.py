import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_generated_graph(graph_data, title="Generated Layout Graph"):
    G = to_networkx(graph_data, to_undirected=True)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(G, node_size=300, node_color='skyblue', font_size=10)
    plt.title(title)
    plt.show()


def inspect_node_features(graph_data):
    print("Generated Node Features:\n", graph_data.x)
