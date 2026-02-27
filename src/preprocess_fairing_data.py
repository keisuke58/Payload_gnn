# preprocess_fairing_data.py
# Converts extracted FEM CSV data to PyTorch Geometric (PyG) Graph Data

import sys
import os
import pandas as pd
import torch
from torch_geometric.data import Data

def load_fem_data(data_dir):
    """Loads nodes and elements CSV files."""
    nodes_path = os.path.join(data_dir, 'nodes.csv')
    elems_path = os.path.join(data_dir, 'elements.csv')
    
    if not os.path.exists(nodes_path) or not os.path.exists(elems_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
        
    df_nodes = pd.read_csv(nodes_path)
    df_elems = pd.read_csv(elems_path)
    
    return df_nodes, df_elems

def build_graph(df_nodes, df_elems):
    """Constructs a PyG Data object from node and element data."""
    
    # 1. Node Features (x, y, z, ux, uy, uz, temp)
    # Map node_id to zero-based index
    node_id_map = {nid: i for i, nid in enumerate(df_nodes['node_id'])}
    
    # Features: [x, y, z, ux, uy, uz, temp]
    # We might want to normalize these later, but for now raw values
    x_features = df_nodes[['x', 'y', 'z', 'ux', 'uy', 'uz', 'temp']].values
    x = torch.tensor(x_features, dtype=torch.float)
    
    # Positions
    pos = torch.tensor(df_nodes[['x', 'y', 'z']].values, dtype=torch.float)
    
    # 2. Edges (Connectivity)
    # We need to convert element connectivity (n1, n2, n3, n4) to edge_index
    # Edges are undirected, so we add (u, v) and (v, u)
    
    src_list = []
    dst_list = []
    
    for _, row in df_elems.iterrows():
        # Nodes in element
        nodes = [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4'])]
        # Filter out 0 (padding)
        nodes = [n for n in nodes if n != 0]
        
        # Create edges between all pairs in the element (clique) or just boundary?
        # Standard FEM-to-Graph usually connects nodes that share an edge in the mesh.
        # For Quad4 (1-2-3-4), edges are 1-2, 2-3, 3-4, 4-1.
        
        num_nodes = len(nodes)
        for i in range(num_nodes):
            u = nodes[i]
            v = nodes[(i + 1) % num_nodes] # Cyclic next
            
            if u in node_id_map and v in node_id_map:
                u_idx = node_id_map[u]
                v_idx = node_id_map[v]
                
                # Add bidirectional
                src_list.append(u_idx)
                dst_list.append(v_idx)
                src_list.append(v_idx)
                dst_list.append(u_idx)
                
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, pos=pos)
    
    return data

def process_dataset(input_dir, output_path):
    print(f"Loading data from {input_dir}...")
    df_nodes, df_elems = load_fem_data(input_dir)
    
    print("Building Graph...")
    data = build_graph(df_nodes, df_elems)
    
    print(f"Graph Info: {data}")
    torch.save(data, output_path)
    print(f"Saved graph to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocess_fairing_data.py <input_dir> <output_path>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    process_dataset(input_dir, output_path)
