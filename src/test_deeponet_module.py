
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_deeponet import DeepONet

def test_deeponet():
    print("Testing DeepONet module...")
    batch_size = 2
    n_sensors = 10
    n_query = 5
    coord_dim = 2
    
    model = DeepONet(sensor_dim=n_sensors, coord_dim=coord_dim)
    
    u_sensors = torch.randn(batch_size, n_sensors)
    y_coords = torch.randn(n_query, coord_dim) # Shared query points
    
    # Test shared query points
    out1 = model(u_sensors, y_coords)
    print(f"Output shape (shared): {out1.shape}")
    assert out1.shape == (batch_size, n_query)
    
    # Test batch query points
    y_coords_batch = torch.randn(batch_size, n_query, coord_dim)
    out2 = model(u_sensors, y_coords_batch)
    print(f"Output shape (batch): {out2.shape}")
    assert out2.shape == (batch_size, n_query)
    
    print("DeepONet module test passed!")

if __name__ == "__main__":
    test_deeponet()
