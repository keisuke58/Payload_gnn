import os
import torch
import pandas as pd
import numpy as np
import itertools
from torch_geometric.data import Data
import enum

# Define Node Types (Customize based on Fairing geometry)
class NodeType(enum.IntEnum):
    INTERNAL = 0
    BOUNDARY = 1
    LOADED = 2
    SIZE = 3

def triangles_to_edges(faces):
    """
    Converts a list of triangular faces to an edge index.
    """
    # Collect edges from triangles
    edges = torch.cat([faces[:, 0:2],
                       faces[:, 1:3],
                       torch.stack([faces[:, 2], faces[:, 0]], dim=1)], dim=0)
    
    # Sort & pack edges to handle undirected graph
    receivers = torch.min(edges, dim=1)[0]
    senders = torch.max(edges, dim=1)[0]
    packed_edges = torch.stack([senders, receivers], dim=1).to(torch.int64)
    
    # Remove duplicates
    unique_edges = torch.unique(packed_edges, dim=0).to(torch.int32)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    
    # Create two-way connectivity (undirected)
    return torch.cat([senders, receivers], dim=0), torch.cat([receivers, senders], dim=0)

def process_fairing_data(data_dir, output_path):
    """
    Reads Abaqus/ANSYS CSV exports and converts them to PyTorch Geometric Data objects.
    
    Expected Input:
    - [design]_nodes.csv: x, y, z, S11, S22, ... nodetype
    - [design]_elements.csv: Element connectivity (node indices)
    """
    data_list = []
    files = os.listdir(data_dir)
    
    # Group files by design name (assuming format: design_nodes.csv, design_elements.csv)
    design_names = set(f.split('_nodes.csv')[0] for f in files if '_nodes.csv' in f)
    
    print(f"Found {len(design_names)} designs to process.")
    
    for design_name in design_names:
        print(f"Processing {design_name}...")
        
        node_file = os.path.join(data_dir, f"{design_name}_nodes.csv")
        elem_file = os.path.join(data_dir, f"{design_name}_elements.csv")
        
        if not os.path.exists(elem_file):
            print(f"Missing element file for {design_name}, skipping.")
            continue
            
        # Read Data
        df_nodes = pd.read_csv(node_file)
        df_elems = pd.read_csv(elem_file)
        
        # --- MESH PROCESSING ---
        # TODO: Adapt for Shell Elements (Triangles/Quads) vs Solid Elements (Tet/Hex)
        # The reference implementation assumes 4-node 3D elements (Tetrahedrons?) 
        # and extracts faces. For Shells, elements ARE faces.
        
        # Example for Shell Elements (if df_elems has columns n1, n2, n3):
        # faces = df_elems[['n1', 'n2', 'n3']].values
        
        # Placeholder for element processing logic
        # For now, we'll assume a 'cells' tensor is created from df_elems
        # This part needs specific logic based on the element type in the CSV
        
        # cells_index = ... 
        # cells = torch.tensor(cells_index, dtype=torch.long)
        
        # --- NODE FEATURES ---
        # Node positions
        pos = df_nodes[['x', 'y', 'z']].values
        mesh_pos = torch.tensor(pos, dtype=torch.float)
        
        # Node Type (Boundary conditions, etc.)
        if 'nodetype' in df_nodes.columns:
            node_type_info = df_nodes['nodetype'].values
            node_type = torch.nn.functional.one_hot(
                torch.tensor(node_type_info, dtype=torch.long), 
                num_classes=NodeType.SIZE
            )
            x = torch.cat([mesh_pos, node_type], dim=-1).float()
        else:
            x = mesh_pos.float()
            
        # --- TARGETS (Stress/Strain) ---
        # Extract stress components if available
        stress_cols = [c for c in df_nodes.columns if c.startswith('S')]
        if stress_cols:
            y = torch.tensor(df_nodes[stress_cols].values, dtype=torch.float)
        else:
            y = None
            
        # --- EDGES ---
        # Generate edges from cells/faces
        # edges_src, edges_dst = triangles_to_edges(torch.tensor(cells_index))
        # edge_index = torch.stack([edges_src, edges_dst], dim=0).long()
        
        # Edge Attributes (Relative position, distance)
        # u_i = mesh_pos[edge_index[0]]
        # u_j = mesh_pos[edge_index[1]]
        # u_ij = u_i - u_j
        # dist = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        # edge_attr = torch.cat([u_ij, dist], dim=-1).float()
        
        # Create Data Object
        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=mesh_pos)
        # data_list.append(data)
        
    # Save processed dataset
    if data_list:
        torch.save(data_list, output_path)
        print(f"Saved processed dataset to {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    # Example usage
    # process_fairing_data("data/raw", "data/processed/fairing_dataset.pt")
    pass
