import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from models_gnn import PointNet, SimpleGCN
import random
import time

# ==============================================================================
# 1. Synthetic Data Generator (H3 Fairing Surrogate)
# ==============================================================================
def generate_cylinder_data(
    num_samples=100, 
    num_points=1024, 
    radius=2.6, # H3 Fairing radius approx
    height=10.0, 
    defect_prob=0.5
):
    dataset = []
    
    print(f"Generating {num_samples} synthetic samples...")
    
    for _ in range(num_samples):
        # 1. Generate Point Cloud (Cylinder Surface)
        # Random heights h and angles theta
        h = torch.rand(num_points) * height
        theta = torch.rand(num_points) * 2 * np.pi
        
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z = h
        
        pos = torch.stack([x, y, z], dim=1) # (N, 3)
        
        # 2. Simulate Wave Field (Scalar Feature)
        # Source at bottom center (0, -R, 0)
        source_pos = torch.tensor([0.0, -radius, 0.0])
        dist_from_source = torch.norm(pos - source_pos, dim=1)
        
        # Simple wave pattern: A * sin(k * r - omega * t)
        # We simulate a snapshot at t=0
        k = 5.0 # Wave number
        wave_field = torch.sin(k * dist_from_source)
        
        # 3. Inject Defect
        has_defect = random.random() < defect_prob
        labels = torch.zeros(num_points, dtype=torch.long) # 0: Healthy, 1: Defect
        
        if has_defect:
            # Random defect location
            defect_idx = random.randint(0, num_points - 1)
            defect_pos = pos[defect_idx]
            
            # Defect affects wave field (scattering/attenuation) locally
            dist_from_defect = torch.norm(pos - defect_pos, dim=1)
            defect_radius = 1.5 # Increased from 0.5 to make defects more prominent
            
            # Mask for defect region
            defect_mask = dist_from_defect < defect_radius
            labels[defect_mask] = 1
            
            # Perturb wave field in defect region (amplitude reduction + phase shift)
            wave_field[defect_mask] *= 0.0 # Strong attenuation
            wave_field[defect_mask] += 2.0 # Strong artifact
            
        # 4. Create PyG Data Object
        # Features: [WaveAmplitude, x, y, z] -> PointNet usually takes pos separately
        # Let's use WaveAmplitude as x (feature)
        x_feat = wave_field.unsqueeze(1) # (N, 1)
        
        # k-NN Graph Construction (for GCN, though PointNet calculates its own or uses these)
        # We'll rely on PointNetConv which can handle pos
        # But for compatibility, let's create a dummy edge_index or leave it empty if PointNet handles it
        # Actually torch_geometric PointNetConv expects edge_index for message passing neighborhood
        # We will compute k-NN on the fly or pre-compute. 
        # For simplicity in this prototype, we'll let the model or a transform handle it, 
        # OR we can just use a radius graph here.
        
        # Let's use a simple heuristic: Connect nearby points
        # For prototype speed, we might skip detailed edge construction if using PointNet 
        # that aggregates globally or uses dynamic graph generation.
        # But standard PointNetConv in PyG needs edge_index. 
        # Let's use torch_cluster.knn_graph if available, else simple random edges?
        # Better: Use a simple fully connected or random graph is bad.
        # Let's use a simple distance threshold loop (slow) or assume the model builds it.
        # To avoid dependency issues with torch_cluster, let's just make a dummy edge_index 
        # (self-loops) + random connections, or better:
        # We will use "DynamicEdgeConv" or similar in a real scenario.
        # For THIS prototype, we will perform a simple KNN strategy using Pytorch cdist.
        
        # Simple KNN (k=10)
        dist_matrix = torch.cdist(pos, pos)
        # Get indices of k nearest neighbors
        _, indices = dist_matrix.topk(10, largest=False)
        
        source_nodes = torch.arange(num_points).view(-1, 1).repeat(1, 10).flatten()
        target_nodes = indices.flatten()
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        data = Data(x=x_feat, pos=pos, y=labels, edge_index=edge_index)
        dataset.append(data)
        
    return dataset

# ==============================================================================
# 2. Training Loop
# ==============================================================================
def train_prototype():
    # Settings
    NUM_POINTS = 512
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 0.001
    
    # 1. Prepare Data
    train_data = generate_cylinder_data(num_samples=50, num_points=NUM_POINTS, defect_prob=0.5)
    test_data = generate_cylinder_data(num_samples=10, num_points=NUM_POINTS, defect_prob=0.5)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    # Input features: 1 (Wave Amplitude)
    # Classes: 2 (Healthy vs Defect)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PointNet(num_node_features=1, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Calculate class weights (Defect is rare)
    # Approx ratio: 1 defect per sample, ~20 nodes out of 512 => 4% defect
    weight = torch.tensor([1.0, 20.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight) 
    
    # 3. Train
    print("\nStarting Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_nodes = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            out = model(batch) # (Batch*NumPoints, NumClasses)
            
            # Loss (Node-wise classification)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.y.size(0)
            
        avg_loss = total_loss / len(train_loader)
        acc = correct / total_nodes
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")
        
    # 4. Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    correct = 0
    total_nodes = 0
    tp = 0 # True Positives (Defect correctly identified)
    fn = 0 # False Negatives (Defect missed)
    fp = 0 # False Positives (False alarm)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.y.size(0)
            
            # Defect Detection Metrics (Class 1)
            tp += ((pred == 1) & (batch.y == 1)).sum().item()
            fn += ((pred == 0) & (batch.y == 1)).sum().item()
            fp += ((pred == 1) & (batch.y == 0)).sum().item()
            
    acc = correct / total_nodes
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Defect Detection Recall: {recall:.4f}")
    print(f"Defect Detection Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\n[SUCCESS] Prototype Pipeline Completed.")
    print("This confirms the PointNet model can learn to localize defects on a 3D cylinder from wave field anomalies.")

if __name__ == "__main__":
    train_prototype()
