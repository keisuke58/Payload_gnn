import os
import requests
import zipfile
import h5py
import json

# Configuration
RECORD_ID = "5105861"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
OUTPUT_DIR = "data/open_guided_waves"
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted")

def get_download_url():
    """Queries Zenodo API to get the direct download URL for the dataset."""
    print(f"Querying Zenodo API: {API_URL}")
    try:
        response = requests.get(API_URL)
        if response.status_code != 200:
            print(f"Failed to query Zenodo API. Status: {response.status_code}")
            return None, None
        
        data = response.json()
        if 'files' not in data:
            print("No files found in Zenodo record.")
            return None, None
            
        # Find the zip file (usually the main dataset)
        for file_info in data['files']:
            filename = file_info['key']
            if filename.endswith('.zip'):
                download_url = file_info['links']['self']
                print(f"Found dataset: {filename}")
                return filename, download_url
        
        # Fallback: just take the first file
        first_file = data['files'][0]
        return first_file['key'], first_file['links']['self']
        
    except Exception as e:
        print(f"Error querying API: {e}")
        return None, None

def download_file(filename, url):
    """Downloads the file from the given URL."""
    if not url:
        return
        
    local_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return local_path

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            print("Download complete.")
            return local_path
        else:
            print(f"Failed to download. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

def extract_zip(zip_path):
    """Extracts the zip file."""
    if not zip_path or not zip_path.endswith('.zip'):
        return False

    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False

def inspect_structure(file_path):
    """Recursively prints the HDF5 group/dataset structure."""
    print(f"\n--- Inspecting {os.path.basename(file_path)} ---")
    
    try:
        with h5py.File(file_path, 'r') as f:
            def print_item(name, obj):
                indent = "  " * (name.count('/') + 1)
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
                    # Print attributes (limit output)
                    for key, val in obj.attrs.items():
                        val_str = str(val)
                        if len(val_str) > 50:
                            val_str = val_str[:50] + "..."
                        print(f"{indent}  Attr: {key} = {val_str}")

            f.visititems(print_item)
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

if __name__ == "__main__":
    filename, url = get_download_url()
    if filename and url:
        file_path = download_file(filename, url)
        if file_path:
            if file_path.endswith('.zip'):
                if extract_zip(file_path):
                    # Inspect H5 files
                    found = False
                    for root, dirs, files in os.walk(EXTRACT_DIR):
                        for file in files:
                            if file.endswith(".h5") or file.endswith(".mat"):
                                inspect_structure(os.path.join(root, file))
                                found = True
                                break 
                        if found: break
                    if not found:
                        print("No .h5/.mat files found in extracted data.")
            elif file_path.endswith('.h5'):
                 inspect_structure(file_path)
