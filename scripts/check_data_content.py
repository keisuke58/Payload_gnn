
import torch
from torch_geometric.data import Data

data_list = torch.load('data/processed_25mm_100/train.pt', weights_only=False)
print(f"Loaded {len(data_list)} samples")
sample = data_list[0]
print("Keys:", sample.keys())
print("x shape:", sample.x.shape)
print("pos shape:", sample.pos.shape if hasattr(sample, 'pos') and sample.pos is not None else "No pos")
print("y shape:", sample.y.shape)
print("edge_index shape:", sample.edge_index.shape)

# Check feature content
print("First node features:", sample.x[0])
