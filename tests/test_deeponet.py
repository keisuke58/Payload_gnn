
import torch
import sys
import os
import unittest

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models_deeponet import DeepONet

class TestDeepONet(unittest.TestCase):
    def test_deeponet_shapes(self):
        batch_size = 2
        n_sensors = 10
        n_query = 5
        coord_dim = 2
        
        model = DeepONet(sensor_dim=n_sensors, coord_dim=coord_dim)
        
        u_sensors = torch.randn(batch_size, n_sensors)
        y_coords = torch.randn(n_query, coord_dim) # Shared query points
        
        # Test shared query points
        out1 = model(u_sensors, y_coords)
        self.assertEqual(out1.shape, (batch_size, n_query))
        
        # Test batch query points
        y_coords_batch = torch.randn(batch_size, n_query, coord_dim)
        out2 = model(u_sensors, y_coords_batch)
        self.assertEqual(out2.shape, (batch_size, n_query))

if __name__ == "__main__":
    unittest.main()
