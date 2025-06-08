import pandas as pd
import os
'''
# Load the sample data provided
simulations_df = pd.read_csv("X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/simulations.csv")
geometry_df = pd.read_csv("X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/geometries.csv")

# Group simulations data by apartment_id
grouped = simulations_df.groupby("apartment_id")

# Partitioning strategy
partitions = []
current_partition = []
current_count = 0
row_limit = 500  # Target partition size

# Collect apartment groups until the total rows reach ~500
for apt_id, group in grouped:
    group_size = len(group)
    if current_count + group_size > row_limit and current_partition:
        partitions.append(current_partition)
        current_partition = []
        current_count = 0
    current_partition.append(apt_id)
    current_count += group_size

# Append the last partition if not empty
if current_partition:
    partitions.append(current_partition)

# Create output directory
output_dir = "X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/partitioned_merged_apartments"
os.makedirs(output_dir, exist_ok=True)

# Merge each partition with geometry and save to CSV
for idx, apartment_ids in enumerate(partitions):
    sim_part = simulations_df[simulations_df['apartment_id'].isin(apartment_ids)]
    geo_part = geometry_df[geometry_df['apartment_id'].isin(apartment_ids)]
    merged_df = pd.merge(geo_part, sim_part, on='apartment_id', suffixes=('_geom', '_sim'))
    merged_df.to_csv(f"{output_dir}/partition_{idx+1}.csv", index=False)

import os
os.listdir(output_dir)  # Show the output files created
'''
# Load the sample data provided
simulations_df = pd.read_csv("X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/simulations.csv")
geometry_df = pd.read_csv("X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/geometries.csv")

# Group simulations data by apartment_id
grouped = simulations_df.groupby("apartment_id")

# Partitioning strategy
partitions = []
current_partition = []
current_count = 0
row_limit = 500  # Target partition size

from tqdm import tqdm

# Collect apartment groups until the total rows reach ~500
for apt_id, group in tqdm(grouped, desc="Partitioning"):
    group_size = len(group)
    if current_count + group_size > row_limit and current_partition:
        partitions.append(current_partition)
        current_partition = []
        current_count = 0
    current_partition.append(apt_id)
    current_count += group_size

# Append the last partition if not empty
if current_partition:
    partitions.append(current_partition)

# Create output directory
output_dir = "X:/Backup/RMIT/Sem_4/Project/Dataset/7070952/partitioned_merged_apartments"
os.makedirs(output_dir, exist_ok=True)

# Track unmatched apartment_ids
sim_ids = set(simulations_df['apartment_id'].unique())
geo_ids = set(geometry_df['apartment_id'].unique())
only_in_sim = sim_ids - geo_ids
only_in_geo = geo_ids - sim_ids

# Merge and save with progress display
for idx, apartment_ids in enumerate(tqdm(partitions, desc="Merging Partitions")):
    sim_part = simulations_df[simulations_df['apartment_id'].isin(apartment_ids)]
    geo_part = geometry_df[geometry_df['apartment_id'].isin(apartment_ids)]
    merged_df = pd.merge(geo_part, sim_part, on='apartment_id', suffixes=('_geom', '_sim'))
    merged_df.to_csv(f"{output_dir}/partition_{idx+1}.csv", index=False)

# Show the output files
output_files_df = pd.DataFrame(os.listdir(output_dir), columns=["Filename"])
import ace_tools as tools; tools.display_dataframe_to_user(name="Output Partitioned Files", dataframe=output_files_df)

# Show unmatched apartment_ids
unmatched_df = pd.DataFrame({
    "Only in Simulations": list(only_in_sim),
    "Only in Geometrics": list(only_in_geo)
})
tools.display_dataframe_to_user(name="Unmatched Apartment IDs", dataframe=unmatched_df)
