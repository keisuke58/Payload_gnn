# models_gnn.py
# PyTorch Geometric models for Fairing Defect Detection

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PointNetConv, global_mean_pool

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling (if graph-level classification)
        # x = global_mean_pool(x, data.batch) 
        
        # Node-level classification (defect segmentation)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

class PointNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = PointNetConv(local_nn=torch.nn.Sequential(
            torch.nn.Linear(num_node_features + 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        ), global_nn=torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64)
        ))
        
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        # pos is usually required for PointNet
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch
        
        x = self.conv1(x, pos, edge_index)
        x = F.relu(x)
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
