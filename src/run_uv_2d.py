import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# from PIL import Image

# =========================================================================
# 1. UV Mapping (Cylindrical Projection)
# =========================================================================
def map_to_uv(df_nodes, width=512, height=512, radius=2600.0, h_max=5000.0):
    """
    Map 3D cylindrical coordinates (x, y, z) to 2D UV image.
    
    Args:
        df_nodes: DataFrame with columns ['x', 'y', 'z', 's11', 's22', 's12', 'dspss', 'defect_label']
        width: Output image width (pixels)
        height: Output image height (pixels)
        radius: Cylinder radius (mm)
        h_max: Cylinder height (mm)
    
    Returns:
        image: (C, H, W) tensor where C is feature channels (s11, s22, s12, dspss)
        mask: (1, H, W) tensor with defect labels (0 or 1)
    """
    x = df_nodes['x'].values
    y = df_nodes['y'].values
    z = df_nodes['z'].values
    
    # Calculate theta (angle in radians)
    theta = np.arctan2(y, x)  # -pi to pi
    
    # Normalize coordinates to [0, 1] range
    u = (theta + np.pi) / (2 * np.pi)  # 0 to 1
    v = z / h_max                      # 0 to 1
    
    # Scale to image dimensions
    u_idx = (u * (width - 1)).astype(int)
    v_idx = (v * (height - 1)).astype(int)
    
    # Clip indices to be safe
    u_idx = np.clip(u_idx, 0, width - 1)
    v_idx = np.clip(v_idx, 0, height - 1)
    
    # Features to map
    features = ['s11', 's22', 's12', 'dspss']
    num_channels = len(features)
    
    # Initialize image and mask
    # We use a "sparse to dense" approach: fill pixels based on nearest node
    # Note: A more sophisticated approach would be interpolation, but for dense FEM mesh, 
    # nearest pixel assignment is a reasonable first step.
    
    image_np = np.zeros((height, width, num_channels), dtype=np.float32)
    mask_np = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    # Populate the image
    # Optimization: This loop is slow in Python. Vectorized approach is better.
    # However, since multiple nodes might map to the same pixel, we average them.
    
    # Create flat indices
    flat_indices = v_idx * width + u_idx
    
    for i, feat in enumerate(features):
        vals = df_nodes[feat].values
        # Using bincount for fast summation of values at same index
        sums = np.bincount(flat_indices, weights=vals, minlength=width*height)
        counts = np.bincount(flat_indices, minlength=width*height)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vals = sums / counts
            avg_vals[counts == 0] = 0
            
        image_np[:, :, i] = avg_vals.reshape(height, width)
        
        if i == 0: # Save counts once
             count_map = counts.reshape(height, width)

    # Process Mask (Label)
    # If ANY node in the pixel is defective, the pixel is defective (conservative)
    labels = df_nodes['defect_label'].values
    max_labels = np.zeros(width * height)
    np.maximum.at(max_labels, flat_indices, labels)
    mask_np = max_labels.reshape(height, width)
    
    # Simple inpainting for empty pixels (holes in the map)
    # For now, we leave them as zeros. In production, we'd use cv2.inpaint
    
    # Transpose to (C, H, W) for PyTorch
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) # (1, H, W)
    
    return image_tensor, mask_tensor

# =========================================================================
# 2. Dataset Class
# =========================================================================
class Fairing2DDataset(Dataset):
    def __init__(self, data_dir, image_size=(512, 512)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.sample_dirs = glob.glob(os.path.join(data_dir, "sample_*"))
        self.sample_dirs.sort() # Ensure deterministic order
        
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        nodes_path = os.path.join(sample_dir, 'nodes.csv')
        
        if not os.path.exists(nodes_path):
            # Return dummy if file missing (shouldn't happen)
            return torch.zeros((4, *self.image_size)), torch.zeros((1, *self.image_size))
            
        df_nodes = pd.read_csv(nodes_path)
        image, mask = map_to_uv(df_nodes, width=self.image_size[0], height=self.image_size[1])
        
        return image, mask

# =========================================================================
# 3. Simple U-Net Model
# =========================================================================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        # Skip connection concatenation
        # Assuming input size is power of 2, no resizing needed
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        out = self.final(d1)
        return out

# =========================================================================
# 4. Main Execution
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_output')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}...")
    # Assume data is organized as dataset_output/sample_0, dataset_output/sample_1, etc.
    # Or subfolders like healthy_baseline. For now, let's just look recursively or specific paths.
    # Adjusting to the observed structure: dataset_output/healthy_baseline might be one sample.
    # Let's generalize to find any folder with nodes.csv
    
    # Quick fix for the specific environment structure
    dataset = Fairing2DDataset(args.data_dir)
    
    if len(dataset) == 0:
        # Fallback for the specific healthy_baseline folder seen in LS
        print("No 'sample_*' directories found. Checking for 'healthy_baseline'...")
        single_sample_path = os.path.join(args.data_dir, "healthy_baseline")
        if os.path.exists(os.path.join(single_sample_path, "nodes.csv")):
             # Mock dataset with 1 sample repeated for testing
             print(f"Found {single_sample_path}. Using as single sample test.")
             dataset.sample_dirs = [single_sample_path] * 10 # Repeat 10 times for fake training
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() # For binary segmentation
    
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training complete.")
    
    # Save model
    os.makedirs('runs/unet', exist_ok=True)
    torch.save(model.state_dict(), 'runs/unet/model.pth')
    print("Model saved to runs/unet/model.pth")
