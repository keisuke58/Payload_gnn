import h5py
import numpy as np
import os

# Assume dataset is extracted here
DATA_DIR = "data/open_guided_waves/extracted"

def inspect_structure(file_path):
    """Recursively prints the HDF5 group/dataset structure."""
    print(f"\n--- Inspecting {os.path.basename(file_path)} ---")
    
    with h5py.File(file_path, 'r') as f:
        def print_item(name, obj):
            indent = "  " * (name.count('/') + 1)
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                # Print attributes
                for key, val in obj.attrs.items():
                    print(f"{indent}  Attr: {key} = {val}")

        f.visititems(print_item)

if __name__ == "__main__":
    # Walk through the directory to find .h5 files
    found = False
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".h5"):
                file_path = os.path.join(root, file)
                inspect_structure(file_path)
                found = True
                break # Inspect only the first one found
        if found:
            break
            
    if not found:
        print(f"No .h5 files found in {DATA_DIR}. Please run the download script first.")
