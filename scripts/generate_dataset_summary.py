import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_dataset_summary(data_dir, output_dir):
    print(f"Data directory: {os.path.abspath(data_dir)}")
    os.makedirs(output_dir, exist_ok=True)
    
    samples = glob.glob(os.path.join(data_dir, "sample_*"))
    print(f"Found {len(samples)} samples.")
    
    metadata_list = []
    
    for i, s in enumerate(samples):
        if i % 10 == 0:
            print(f"Processing {i}/{len(samples)}: {s}")
            
        meta_file = os.path.join(s, "metadata.csv")
        if os.path.exists(meta_file):
            try:
                df = pd.read_csv(meta_file)
                if not df.empty and 'key' in df.columns and 'value' in df.columns:
                    # Metadata is key-value pairs
                    row = dict(zip(df['key'], df['value']))
                    
                    # Convert numeric fields
                    for k in ['theta_deg', 'z_center', 'radius']:
                        if k in row:
                            try:
                                row[k] = float(row[k])
                            except:
                                pass
                    
                    row['sample_id'] = os.path.basename(s)
                    
                    # Check if nodes.csv has valid displacement
                    nodes_file = os.path.join(s, "nodes.csv")
                    if os.path.exists(nodes_file):
                        try:
                            nodes_df = pd.read_csv(nodes_file)
                            if 'ux' in nodes_df.columns:
                                max_ux = nodes_df['ux'].abs().max()
                                row['max_ux'] = max_ux
                                row['valid'] = max_ux > 0.001
                            else:
                                row['valid'] = False
                        except Exception as e:
                            print(f"Error reading nodes {s}: {e}")
                            row['valid'] = False
                    else:
                        row['valid'] = False
                    
                    metadata_list.append(row)
                else:
                    # Fallback for old format if any
                    pass
            except Exception as e:
                print(f"Error reading metadata {s}: {e}")
    
    if not metadata_list:
        print("No metadata found.")
        return

    df_meta = pd.DataFrame(metadata_list)
    
    # Filter valid
    df_valid = df_meta[df_meta.get('valid', False) == True]
    print(f"Valid samples: {len(df_valid)} / {len(df_meta)}")
    
    if df_valid.empty:
        print("No valid samples found to plot.")
        return

    # 1. Spatial Distribution (Theta vs Z)
    try:
        plt.figure(figsize=(10, 6))
        # Ensure columns exist
        if 'z_center' in df_valid.columns and 'theta_deg' in df_valid.columns:
            sc = plt.scatter(df_valid['z_center'], df_valid['theta_deg'], 
                        s=df_valid['radius']*2, alpha=0.6, c='blue', edgecolors='k')
            plt.xlabel('Z Center (mm)')
            plt.ylabel('Theta (deg)')
            plt.title(f'Defect Spatial Distribution (N={len(df_valid)})')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Legend for sizes
            sizes = [20, 50, 100, 200]
            labels = [f'R={r}mm' for r in sizes]
            handles = [plt.scatter([], [], s=r*2, c='blue', alpha=0.6, edgecolors='k') for r in sizes]
            plt.legend(handles, labels, title="Defect Size", loc="upper right")
            
            plt.tight_layout()
            out_path = os.path.join(output_dir, "01_spatial_distribution.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")
        else:
            print("Missing z_center or theta_deg")
    except Exception as e:
        print(f"Error plotting spatial: {e}")
    
    # 2. Radius Distribution
    try:
        plt.figure(figsize=(8, 5))
        if 'radius' in df_valid.columns:
            plt.hist(df_valid['radius'], bins=15, edgecolor='black', alpha=0.7)
            plt.xlabel('Defect Radius (mm)')
            plt.ylabel('Count')
            plt.title('Defect Size Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(output_dir, "02_radius_distribution.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")
        else:
            print("Missing radius")
    except Exception as e:
        print(f"Error plotting radius: {e}")
    
    # 3. Max Displacement vs Radius
    try:
        plt.figure(figsize=(8, 6))
        if 'radius' in df_valid.columns and 'max_ux' in df_valid.columns:
            plt.scatter(df_valid['radius'], df_valid['max_ux'], alpha=0.7)
            plt.xlabel('Defect Radius (mm)')
            plt.ylabel('Max Displacement |ux| (mm)')
            plt.title('Defect Size vs Max Displacement')
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(output_dir, "03_size_vs_displacement.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")
        else:
            print("Missing radius or max_ux")
    except Exception as e:
        print(f"Error plotting displacement: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_output", help="Data directory")
    parser.add_argument("--output", default="wiki_repo/images/dataset_summary", help="Output directory")
    args = parser.parse_args()
    
    generate_dataset_summary(args.data, args.output)
