from graph_decoder import generate_layout_graph
from visualize_graph import visualize_generated_graph, inspect_node_features

# Parameters must match training setup
LATENT_DIM = 32
NODE_FEATURE_DIM = 33
SAMPLE_SIZE = 8

# Generate a layout graph
generated_graph = generate_layout_graph(z_dim=LATENT_DIM, out_node_dim=NODE_FEATURE_DIM, sample_size=SAMPLE_SIZE)

# Visualize the layout graph
visualize_generated_graph(generated_graph, title="Sampled Layout Graph")

# Print node features for inspection
inspect_node_features(generated_graph)
