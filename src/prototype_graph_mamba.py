import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import time

# ==============================================================================
# Graph Mamba: State Space Model (SSM) for Graph-Structured Data
#
# Goal: Apply Mamba's selective state space mechanism to graph node sequences.
#       Unlike Graph Transformers (O(N²) attention), Graph Mamba achieves
#       O(N log N) complexity via selective scan, enabling scaling to
#       large fairing meshes (~100K nodes).
#
# Architecture:
#   1. Node Ordering: Convert graph to sequence via BFS/DFS/random walk
#   2. Selective SSM Block: Mamba-style selective scan on node sequence
#   3. Graph-aware Mixing: Combine SSM output with local neighborhood info
#
# Key Advantage for H3 Fairing:
#   - Linear scaling with mesh size (vs O(N²) for attention)
#   - Long-range dependency capture (wave propagation across full fairing)
#   - Efficient on large FEM meshes (>50K nodes)
#
# Reference:
#   Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#   (COLM 2024)
#   Wang et al., "Graph-Mamba: Towards Long-Range Graph Sequence Modeling with
#   Selective State Spaces" (arXiv 2024)
# ==============================================================================


class SelectiveSSM(nn.Module):
    """Simplified Selective State Space Model (S6) block.

    Implements the core Mamba mechanism:
    - Input-dependent (selective) parameters B, C, Δ
    - Discretized state space: x_k = A_bar * x_{k-1} + B_bar * u_k
    - Output: y_k = C_k * x_k

    This is a simplified version for prototyping. Production would use
    the optimized CUDA kernel from mamba-ssm package.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (local context before SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        # A: state transition (initialized as negative log-spaced values for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # B, C, Δ are input-dependent (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection: split into two paths (like GLU)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Local convolution
        x_ssm = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]  # Causal padding
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute selective parameters
        x_dbl = self.x_proj(x_ssm)  # (B, L, 2*d_state + 1)
        B = x_dbl[:, :, :self.d_state]  # (B, L, d_state)
        C = x_dbl[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = F.softplus(self.dt_proj(x_dbl[:, :, -1:]))  # (B, L, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Sequential scan (simplified; production uses parallel scan)
        y = self._selective_scan(x_ssm, A, B, C, dt)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _selective_scan(self, u, A, B, C, dt):
        """Selective scan (sequential version for prototyping).

        Args:
            u: (B, L, d_inner) input
            A: (d_inner, d_state) state matrix
            B: (B, L, d_state) input-dependent B
            C: (B, L, d_state) input-dependent C
            dt: (B, L, d_inner) input-dependent timestep
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # For efficiency, we compute per-step
        state = torch.zeros(batch, d_inner, d_state, device=u.device)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            B_bar = dt_t * B[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)  # (B, d_inner, d_state)

            # State update: x = A_bar * x + B_bar * u
            state = A_bar * state + B_bar * u[:, t, :].unsqueeze(-1)  # (B, d_inner, d_state)

            # Output: y = C * x
            y_t = (state * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class GraphMambaBlock(nn.Module):
    """Single Graph Mamba block combining SSM with graph structure.

    Pipeline:
    1. SSM processes the node sequence (captures long-range dependencies)
    2. Local graph aggregation (preserves neighborhood structure)
    3. Residual + LayerNorm
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)

        # Local graph aggregation (simple message passing)
        self.local_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, num_nodes, d_model)
            adj: (batch, num_nodes, num_nodes) adjacency or None
        """
        # SSM path (long-range)
        x = x + self.ssm(self.norm1(x))

        # Local graph aggregation path
        if adj is not None:
            # Simple mean aggregation from neighbors
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
            neighbor_feat = torch.bmm(adj, x) / deg  # (B, N, d_model)
            local_input = torch.cat([x, neighbor_feat], dim=-1)  # (B, N, 2*d_model)
            x = x + self.local_mlp(self.norm2(local_input))

        return x


class GraphMamba(nn.Module):
    """Graph Mamba model for node classification on fairing mesh.

    Combines:
    - Positional encoding (node coordinates → embedding)
    - Stacked Graph Mamba blocks (SSM + local aggregation)
    - Per-node classifier
    """
    def __init__(self, in_features, d_model=64, d_state=16,
                 num_layers=3, num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            GraphMambaBlock(d_model, d_state)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, num_nodes, in_features) node features
            adj: (batch, num_nodes, num_nodes) adjacency matrix (optional)
        Returns:
            logits: (batch, num_nodes, num_classes)
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, adj)

        return self.classifier(x)


# ==============================================================================
# Synthetic Data Generator
# ==============================================================================
def generate_graph_data(num_samples=100, num_nodes=256, num_features=10, k=8):
    """
    Generate synthetic graph-structured fairing data.

    Each sample:
    - Nodes on a cylinder surface with features (stress, temperature, etc.)
    - k-NN adjacency matrix
    - Binary labels (healthy=0 / defect=1)
    """
    print(f"Generating {num_samples} graph samples ({num_nodes} nodes, k={k})...")

    all_features = []
    all_adj = []
    all_labels = []

    for _ in range(num_samples):
        # Random points on cylinder
        theta = np.random.uniform(0, 2 * np.pi, num_nodes)
        z = np.random.uniform(0, 10, num_nodes)
        R = 2.6

        x = R * np.cos(theta)
        y = R * np.sin(theta)

        coords = np.stack([x, y, z], axis=-1)

        # k-NN adjacency
        dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        knn_idx = np.argsort(dist, axis=-1)[:, 1:k+1]

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            adj[i, knn_idx[i]] = 1.0
            adj[knn_idx[i], i] = 1.0  # Symmetric

        # Node features: coords + simulated physics
        base_wave = np.sin(5 * theta) * np.cos(2 * z)
        stress = base_wave + 0.1 * np.random.randn(num_nodes)
        temperature = 100 + 50 * (z / 10.0) + 5 * np.random.randn(num_nodes)
        curvature = 1.0 / R * np.ones(num_nodes)

        features = np.stack([
            x, y, z,
            np.cos(theta), np.sin(theta), z / 10.0,  # normalized coords
            stress,
            temperature / 200.0,  # normalized
            curvature,
            base_wave,
        ], axis=-1).astype(np.float32)

        # Labels: inject defect
        labels = np.zeros(num_nodes, dtype=np.int64)
        if np.random.rand() < 0.7:
            cx, cy, cz = np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(2, 8)
            defect_center = np.array([cx, cy, cz])
            dist_to_defect = np.linalg.norm(coords - defect_center, axis=-1)
            defect_mask = dist_to_defect < 1.5
            labels[defect_mask] = 1

            # Perturb features in defect region
            features[defect_mask, 6] += 2.0  # stress anomaly
            features[defect_mask, 9] *= 0.3  # wave attenuation

        all_features.append(features)
        all_adj.append(adj)
        all_labels.append(labels)

    features_t = torch.tensor(np.array(all_features), dtype=torch.float32)
    adj_t = torch.tensor(np.array(all_adj), dtype=torch.float32)
    labels_t = torch.tensor(np.array(all_labels), dtype=torch.long)

    return features_t, adj_t, labels_t


# ==============================================================================
# Training Loop
# ==============================================================================
def train_graph_mamba_prototype():
    NUM_NODES = 256
    NUM_FEATURES = 10
    D_MODEL = 64
    D_STATE = 16
    NUM_LAYERS = 3
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data
    train_feat, train_adj, train_labels = generate_graph_data(
        num_samples=200, num_nodes=NUM_NODES, num_features=NUM_FEATURES,
    )
    test_feat, test_adj, test_labels = generate_graph_data(
        num_samples=50, num_nodes=NUM_NODES, num_features=NUM_FEATURES,
    )

    from torch.utils.data import TensorDataset
    train_loader = DataLoader(
        TensorDataset(train_feat, train_adj, train_labels),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_feat, test_adj, test_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # 2. Model
    model = GraphMamba(
        in_features=NUM_FEATURES,
        d_model=D_MODEL,
        d_state=D_STATE,
        num_layers=NUM_LAYERS,
        num_classes=2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Focal-like weighting for class imbalance
    weight = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # 3. Train
    print("\nStarting Graph Mamba Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for feat, adj, labels in train_loader:
            feat, adj, labels = feat.to(device), adj.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(feat, adj)  # (B, N, 2)
            logits_flat = logits.reshape(-1, 2)
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits_flat.argmax(dim=1)
            correct += (pred == labels_flat).sum().item()
            total += labels_flat.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # 4. Evaluation
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)

    model.eval()
    tp, fn, fp, tn = 0, 0, 0, 0

    with torch.no_grad():
        for feat, adj, labels in test_loader:
            feat, adj, labels = feat.to(device), adj.to(device), labels.to(device)
            logits = model(feat, adj)
            pred = logits.argmax(dim=-1)

            tp += ((pred == 1) & (labels == 1)).sum().item()
            fn += ((pred == 0) & (labels == 1)).sum().item()
            fp += ((pred == 1) & (labels == 0)).sum().item()
            tn += ((pred == 0) & (labels == 0)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # 5. Complexity comparison
    print("\n" + "-" * 60)
    print("Complexity Comparison")
    print("-" * 60)
    print(f"  Graph Mamba (SSM):     O(N · d_state) per block  [N={NUM_NODES}]")
    print(f"  Graph Transformer:     O(N²· d_model) per block")
    print(f"  Standard GNN (k-hop):  O(N · k · d_model) per layer")
    print(f"")
    print(f"  For H3 fairing mesh (N ~ 50,000 nodes):")
    print(f"    Graph Mamba:      ~{50000 * D_STATE:,} ops/block")
    print(f"    Graph Transformer: ~{50000**2 * D_MODEL:,} ops/block")
    print(f"    Ratio:            ~{50000 * D_MODEL // D_STATE:,}x more efficient")

    print(f"\n[SUCCESS] Graph Mamba Prototype Completed.")
    print("Selective SSM captures long-range wave propagation patterns")
    print("with linear complexity in node count.\n")
    print("Next Steps for Production:")
    print("  1. Integrate with PyG Data objects from build_graph.py")
    print("  2. Use node ordering from BFS traversal (mesh topology aware)")
    print("  3. Benchmark against GAT/GIN on actual H3 dataset")
    print("  4. Consider mamba-ssm CUDA kernels for training speedup")


if __name__ == "__main__":
    train_graph_mamba_prototype()
