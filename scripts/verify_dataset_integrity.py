
import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.build_graph import build_curvature_graph

def verify_dataset(dataset_dir):
    print(f"Verifying dataset in: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return

    sample_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sample_') or d == 'healthy_baseline']
    sample_dirs.sort()
    
    print(f"Found {len(sample_dirs)} samples.")
    
    valid_samples = 0
    invalid_samples = 0
    stats = []

    for sample_name in tqdm(sample_dirs):
        sample_path = os.path.join(dataset_dir, sample_name)
        nodes_path = os.path.join(sample_path, 'nodes.csv')
        elems_path = os.path.join(sample_path, 'elements.csv')
        meta_path = os.path.join(sample_path, 'metadata.csv')
        
        # Check file existence
        if not (os.path.exists(nodes_path) and os.path.exists(elems_path)):
            print(f"[FAIL] {sample_name}: Missing CSV files")
            invalid_samples += 1
            continue
            
        try:
            # Load Data
            df_nodes = pd.read_csv(nodes_path)
            df_elems = pd.read_csv(elems_path)
            
            # Check Metadata consistency
            meta_defect_count = 0
            if os.path.exists(meta_path):
                df_meta = pd.read_csv(meta_path, index_col=0)
                if 'n_defect_nodes' in df_meta.index:
                    meta_defect_count = int(df_meta.loc['n_defect_nodes'].values[0])
            
            # Check Node Labels
            if 'defect_label' not in df_nodes.columns:
                print(f"[FAIL] {sample_name}: 'defect_label' column missing in nodes.csv")
                invalid_samples += 1
                continue

            # defect_label: 0=healthy, 1-7=defect types (multi-class supported)
            actual_defect_count = (df_nodes['defect_label'] != 0).sum()
            
            # Check for NaNs
            if df_nodes.isnull().values.any():
                print(f"[WARN] {sample_name}: NaN values found in nodes.csv")
            
            # Try building graph
            try:
                # Disable geodesic for speed in verification
                data = build_curvature_graph(df_nodes, df_elems, compute_geodesic=False, verbose=False)
                
                # Check graph properties
                if data.num_nodes == 0:
                    print(f"[FAIL] {sample_name}: Graph has 0 nodes")
                    invalid_samples += 1
                    continue
                    
                if data.x.shape[1] == 0:
                    print(f"[FAIL] {sample_name}: Node features empty")
                    invalid_samples += 1
                    continue
                    
            except Exception as e:
                print(f"[FAIL] {sample_name}: Graph build error: {e}")
                invalid_samples += 1
                continue

            # Consistency Check
            is_healthy = (sample_name == 'healthy_baseline')
            if not is_healthy and actual_defect_count == 0:
                 # It's possible for very small defects to miss nodes if mesh is coarse, but for 25mm it should hit.
                 # However, let's just log it.
                 pass
            
            stats.append({
                'name': sample_name,
                'nodes': len(df_nodes),
                'elems': len(df_elems),
                'defects_meta': meta_defect_count,
                'defects_actual': actual_defect_count,
                'consistent': (meta_defect_count == actual_defect_count)
            })
            
            valid_samples += 1
            
        except Exception as e:
            print(f"[FAIL] {sample_name}: Unexpected error: {e}")
            invalid_samples += 1

    print("-" * 50)
    print(f"Summary for {dataset_dir}")
    print(f"Total: {len(sample_dirs)}")
    print(f"Valid: {valid_samples}")
    print(f"Invalid: {invalid_samples}")
    
    if stats:
        df_stats = pd.DataFrame(stats)
        print("\nDefect Node Counts (Top 10):")
        print(df_stats[['name', 'defects_meta', 'defects_actual', 'consistent']].head(10))
        
        zero_defects = df_stats[df_stats['defects_actual'] == 0]
        if not zero_defects.empty:
            print(f"\nSamples with 0 defect nodes ({len(zero_defects)}):")
            print(zero_defects['name'].tolist())
            
        inconsistent = df_stats[~df_stats['consistent']]
        if not inconsistent.empty:
            print(f"\nInconsistent Metadata vs Actual ({len(inconsistent)}):")
            print(inconsistent[['name', 'defects_meta', 'defects_actual']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
    args = parser.parse_args()
    
    verify_dataset(args.dataset_dir)
