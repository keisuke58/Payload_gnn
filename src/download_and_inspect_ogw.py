import os
import requests
import zipfile
import h5py
import numpy as np

# Configuration
DATASET_URL = "https://zenodo.org/records/5105861/files/Carbon_Fiber_Composite_Plate_with_Omega_Stringer.zip?download=1"
OUTPUT_DIR = "data/open_guided_waves"
ZIP_FILE = os.path.join(OUTPUT_DIR, "dataset.zip")
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted")

def download_dataset():
    """Downloads the Open Guided Waves dataset (CFRP Plate with Omega Stringer)."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if os.path.exists(ZIP_FILE):
        print(f"Dataset already downloaded at {ZIP_FILE}")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL, stream=True)
    if response.status_code == 200:
        with open(ZIP_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def extract_dataset():
    """Extracts the downloaded zip file."""
    if not os.path.exists(ZIP_FILE):
        print("Zip file not found.")
        return

    print(f"Extracting {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")

def inspect_hdf5_file(file_path):
    """Inspects the structure of an HDF5 file."""
    print(f"\nInspecting: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print("Keys:", list(f.keys()))
            
            # Recursively print structure (limited depth)
            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    Attr: {key} = {val}")
            
            f.visititems(print_attrs)
            
            # Example: Accessing specific data (adjust based on actual structure)
            # if 'data' in f:
            #     print("Data shape:", f['data'].shape)
            
    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    
    # Find and inspect the first H5 file
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".h5") or file.endswith(".mat"): # Sometimes MAT files are HDF5
                inspect_hdf5_file(os.path.join(root, file))
                break # Inspect only one for now
