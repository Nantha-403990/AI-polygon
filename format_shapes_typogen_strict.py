
import pandas as pd

def format_shapes_typogen_strict(
    node_feature_csv,
    node_labels_csv,
    output_txt_path,
    rooms_per_floor=16,
    floors_per_layout=1
):
    df_labels = pd.read_csv(node_labels_csv)
    labels = df_labels['layout_area_type'].astype(int).tolist()

    shape_lines = []
    total_nodes = len(labels)
    block_size = rooms_per_floor * floors_per_layout
    num_blocks = total_nodes // block_size

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block_labels = labels[start:end]

        line = "4|0|90|180|270|"
        line += f"{floors_per_layout}|{rooms_per_floor}@"

        for j in range(0, len(block_labels), rooms_per_floor):
            floor_labels = block_labels[j:j+rooms_per_floor]
            for lbl in floor_labels:
                # Strict formatting with label_id + 3 placeholders, each ending with '|'
                line += f"{lbl},|0,|0,|0,|#"
            line = line.rstrip("#") + "#$"  # Floor delimiter

        shape_lines.append(line)

    with open(output_txt_path, "w") as f:
        f.write("\n".join(shape_lines))

if __name__ == "__main__":
    format_shapes_typogen_strict(
        node_feature_csv="Data/Output/generated_node_features.csv",
        node_labels_csv="Data/Output/node_labels_filtered.csv",
        output_txt_path="Data/Output/shapes.txt",
        rooms_per_floor=40,
        floors_per_layout=1
    )
