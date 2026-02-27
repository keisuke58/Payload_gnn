import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointNetLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (Batch, N, In_Channels)
        batch_size, num_points, _ = x.size()
        x = x.view(-1, x.size(-1))
        x = self.mlp(x)
        x = x.view(batch_size, num_points, -1)
        return x

class PointTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(PointTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x: (Batch, N, Dim)
        # Self-Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class SimplePointTransformer(nn.Module):
    def __init__(self, in_channels=10, num_classes=2, embed_dim=64):
        super(SimplePointTransformer, self).__init__()
        
        # Input Embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer Blocks
        self.transformer1 = PointTransformerBlock(embed_dim)
        self.transformer2 = PointTransformerBlock(embed_dim)
        
        # Global Feature Aggregation (Max Pooling)
        # In a segmentation task (node classification), we combine local + global
        
        # Classification Head (per point)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), # Concatenate local + global
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (Batch, N, In_Channels)
        # Embed points
        local_feat = self.embedding(x) # (B, N, Embed)
        
        # Apply Transformer
        local_feat = self.transformer1(local_feat)
        local_feat = self.transformer2(local_feat) # (B, N, Embed)
        
        # Global Feature
        global_feat = torch.max(local_feat, dim=1)[0] # (B, Embed)
        
        # Expand global feature to each point
        global_feat_expanded = global_feat.unsqueeze(1).repeat(1, x.size(1), 1) # (B, N, Embed)
        
        # Concatenate
        combined = torch.cat([local_feat, global_feat_expanded], dim=-1) # (B, N, Embed*2)
        
        # Classify
        out = self.classifier(combined) # (B, N, Num_Classes)
        
        # Reorder for CrossEntropyLoss: (B, C, N)
        return out.permute(0, 2, 1)

# Usage Example / Test
if __name__ == "__main__":
    # Batch=2, Points=1000, Features=10
    dummy_input = torch.randn(2, 1000, 10) 
    model = SimplePointTransformer(in_channels=10, num_classes=2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (2, 2, 1000)
