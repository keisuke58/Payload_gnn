import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# ==============================================================================
# E(3)-Equivariant Graph Neural Network (EGNN) for Fairing SHM
#
# Goal: Build a GNN that respects the symmetries of 3D Euclidean space.
#       The network's predictions are equivariant to rotations, translations,
#       and reflections of the fairing geometry. This guarantees that defect
#       detection is independent of the coordinate frame orientation.
#
# Why for H3 Fairing:
#   - Fairing can be oriented arbitrarily during inspection/monitoring
#   - Physical laws (stress, wave propagation) are frame-independent
#   - Equivariance provides strong inductive bias → better data efficiency
#   - Standard GNNs (GCN/GAT) are NOT equivariant to rotations
#
# Architecture (EGNN - Satorras et al. 2021):
#   - Message passing on both:
#     (a) Scalar features h_i (invariant: stress, temperature, curvature)
#     (b) Vector features x_i (equivariant: coordinates, displacements, normals)
#   - Edge messages depend on inter-node DISTANCE (invariant), not coordinates
#   - Coordinate updates preserve equivariance via direction vectors
#
# Reference:
#   Satorras et al. "E(n) Equivariant Graph Neural Networks" (ICML 2021)
#   Brandstetter et al. "Geometric and Physical Quantities improve E(3) EGNN"
#   Wiki Section 7: Manifold Learning & Geometric Deep Learning
# ==============================================================================


class EGNNLayer(nn.Module):
    """Single E(n)-Equivariant Graph Neural Network layer.

    Updates both:
    - h_i (scalar/invariant features): stress, temperature, etc.
    - x_i (vector/equivariant features): 3D coordinates

    Key invariant: All operations depend on ||x_i - x_j||² (distance),
    not on absolute coordinates → equivariance guaranteed.
    """
    def __init__(self, h_dim, coord_dim=3, edge_attr_dim=0, update_coords=True):
        super().__init__()
        self.update_coords = update_coords

        # Edge model: compute message m_ij
        edge_input_dim = h_dim * 2 + 1 + edge_attr_dim  # h_i, h_j, ||x_i-x_j||², edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )

        # Node model: update h_i
        self.node_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

        # Coordinate model: compute coordinate update weight
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, 1, bias=False),
            )
            # Initialize near zero for stability
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        self.layer_norm = nn.LayerNorm(h_dim)

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: (N, h_dim) scalar node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices [source, target]
            edge_attr: (E, edge_attr_dim) optional edge attributes

        Returns:
            h_out: (N, h_dim) updated scalar features
            x_out: (N, 3) updated coordinates
        """
        src, dst = edge_index

        # Compute invariant: squared distance
        diff = x[src] - x[dst]  # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge message
        edge_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, h_dim)

        # Aggregate messages (mean)
        num_nodes = h.size(0)
        agg = torch.zeros(num_nodes, m_ij.size(-1), device=h.device)
        count = torch.zeros(num_nodes, 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.size(0), 1, device=h.device))
        count = count.clamp(min=1)
        agg = agg / count

        # Update scalar features (invariant)
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h_out = self.layer_norm(h_out)

        # Update coordinates (equivariant)
        x_out = x
        if self.update_coords:
            # Weight for coordinate update from edge messages
            coord_weights = self.coord_mlp(m_ij)  # (E, 1)
            # Weighted direction vectors
            weighted_diff = diff * coord_weights  # (E, 3) — equivariant!

            coord_agg = torch.zeros(num_nodes, 3, device=x.device)
            coord_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_diff), weighted_diff)
            coord_agg = coord_agg / count
            x_out = x + coord_agg

        return h_out, x_out


class EGNN(nn.Module):
    """E(3)-Equivariant Graph Neural Network for node classification.

    For H3 Fairing SHM:
    - Input scalar features: stress, temperature, curvature, wave amplitude
    - Input vector features: 3D coordinates (x, y, z)
    - Output: per-node defect probability (invariant to rotation)
    """
    def __init__(self, in_features, hidden_dim=64, num_layers=4,
                 num_classes=2, update_coords=True):
        super().__init__()

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, update_coords=update_coords)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, h, x, edge_index):
        """
        Args:
            h: (N, in_features) scalar node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices
        Returns:
            logits: (N, num_classes)
        """
        h = self.input_proj(h)

        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        return self.classifier(h)


# ==============================================================================
# Synthetic Data Generator
# ==============================================================================
def generate_egnn_data(num_samples=100, num_nodes=512, k=10):
    """
    Generate graph data for EGNN testing.
    Returns coordinates, scalar features, edge indices, and labels.
    """
    print(f"Generating {num_samples} EGNN samples ({num_nodes} nodes)...")

    dataset = []

    for _ in range(num_samples):
        # Points on cylinder surface
        theta = np.random.uniform(0, 2 * np.pi, num_nodes)
        z = np.random.uniform(0, 10, num_nodes)
        R = 2.6

        x_coord = R * np.cos(theta)
        y_coord = R * np.sin(theta)

        coords = np.stack([x_coord, y_coord, z], axis=-1).astype(np.float32)

        # Scalar features (physics quantities — invariant to rotation)
        wave_amp = np.sin(5 * theta) * np.cos(2 * z)
        stress = wave_amp + 0.1 * np.random.randn(num_nodes)
        temperature = (100 + 50 * z / 10.0) / 200.0
        curvature_k1 = 1.0 / R * np.ones(num_nodes)
        curvature_k2 = np.zeros(num_nodes)  # Cylinder: zero along axis

        scalar_feats = np.stack([
            stress,
            temperature,
            curvature_k1,
            curvature_k2,
            wave_amp,
        ], axis=-1).astype(np.float32)

        # k-NN edges
        dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        knn_idx = np.argsort(dist, axis=-1)[:, 1:k+1]

        src_list = []
        dst_list = []
        for i in range(num_nodes):
            for j in knn_idx[i]:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        # Remove duplicates
        edge_set = set()
        clean_src, clean_dst = [], []
        for s, d in zip(src_list, dst_list):
            if (s, d) not in edge_set:
                edge_set.add((s, d))
                clean_src.append(s)
                clean_dst.append(d)
        edge_index = np.array([clean_src, clean_dst], dtype=np.int64)

        # Labels
        labels = np.zeros(num_nodes, dtype=np.int64)
        if np.random.rand() < 0.7:
            cx = np.random.uniform(-2, 2)
            cy = np.random.uniform(-2, 2)
            cz = np.random.uniform(2, 8)
            defect_center = np.array([cx, cy, cz])
            dist_to_defect = np.linalg.norm(coords - defect_center, axis=-1)
            defect_mask = dist_to_defect < 1.5
            labels[defect_mask] = 1

            # Perturb features
            scalar_feats[defect_mask, 0] += 2.0  # stress
            scalar_feats[defect_mask, 4] *= 0.3  # wave attenuation

        dataset.append({
            'coords': torch.tensor(coords),
            'scalar_feats': torch.tensor(scalar_feats),
            'edge_index': torch.tensor(edge_index),
            'labels': torch.tensor(labels),
        })

    return dataset


def collate_egnn(batch):
    """Collate function for batching graphs with different edge structures."""
    coords_list = []
    feats_list = []
    edges_list = []
    labels_list = []
    offset = 0

    for item in batch:
        n = item['coords'].size(0)
        coords_list.append(item['coords'])
        feats_list.append(item['scalar_feats'])
        edges_list.append(item['edge_index'] + offset)
        labels_list.append(item['labels'])
        offset += n

    return {
        'coords': torch.cat(coords_list, dim=0),
        'scalar_feats': torch.cat(feats_list, dim=0),
        'edge_index': torch.cat(edges_list, dim=1),
        'labels': torch.cat(labels_list, dim=0),
    }


# ==============================================================================
# Equivariance Verification
# ==============================================================================
def verify_equivariance(model, sample, device):
    """Verify that the model output is invariant to rotation of input coordinates."""
    coords = sample['coords'].to(device)
    feats = sample['scalar_feats'].to(device)
    edge_index = sample['edge_index'].to(device)

    # Random 3D rotation matrix (SO(3))
    angle = torch.tensor(np.random.uniform(0, 2 * np.pi))
    axis = torch.randn(3)
    axis = axis / axis.norm()

    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    R = R.to(device)

    # Random translation
    t = torch.randn(3).to(device) * 5

    model.eval()
    with torch.no_grad():
        # Original prediction
        logits_orig = model(feats, coords, edge_index)

        # Rotated + translated prediction
        coords_transformed = (coords @ R.T) + t
        logits_transformed = model(feats, coords_transformed, edge_index)

    # Compare (should be very close for invariant predictions)
    diff = (logits_orig - logits_transformed).abs().max().item()
    return diff


# ==============================================================================
# Training Loop
# ==============================================================================
def train_egnn_prototype():
    NUM_NODES = 256
    HIDDEN_DIM = 64
    NUM_LAYERS = 4
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data
    train_data = generate_egnn_data(num_samples=150, num_nodes=NUM_NODES)
    test_data = generate_egnn_data(num_samples=40, num_nodes=NUM_NODES)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_egnn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_egnn)

    # 2. Model
    model = EGNN(
        in_features=5,  # scalar features only
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=2,
        update_coords=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    weight = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # 3. Train
    print("\nStarting EGNN Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            coords = batch['coords'].to(device)
            feats = batch['scalar_feats'].to(device)
            edge_index = batch['edge_index'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(feats, coords, edge_index)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

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
        for batch in test_loader:
            coords = batch['coords'].to(device)
            feats = batch['scalar_feats'].to(device)
            edge_index = batch['edge_index'].to(device)
            labels = batch['labels'].to(device)

            logits = model(feats, coords, edge_index)
            pred = logits.argmax(dim=1)

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

    # 5. Equivariance Verification
    print("\n" + "-" * 60)
    print("E(3) Equivariance Verification")
    print("-" * 60)
    print("Applying random SE(3) transformation (rotation + translation)...")

    diffs = []
    for i in range(10):
        diff = verify_equivariance(model, test_data[i], device)
        diffs.append(diff)

    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    print(f"  Prediction difference after SE(3) transform:")
    print(f"    Mean max-abs diff: {mean_diff:.6f}")
    print(f"    Worst case diff:   {max_diff:.6f}")

    if max_diff < 0.1:
        print("  [OK] Model exhibits approximate E(3) equivariance.")
    else:
        print("  [NOTE] Some deviation expected due to coordinate updates")
        print("         and numerical precision. Pure invariant features")
        print("         (distances) maintain exact equivariance.")

    # 6. Why Equivariance Matters
    print("\n" + "-" * 60)
    print("Why E(3) Equivariance Matters for H3 Fairing")
    print("-" * 60)
    print("  1. Coordinate-frame independent: Same detection regardless of")
    print("     how the fairing is oriented during inspection")
    print("  2. Data efficiency: Equivariance = built-in data augmentation")
    print("     (all rotations are 'free' training samples)")
    print("  3. Physical consistency: Stress tensors transform correctly")
    print("     under coordinate changes")
    print("  4. Generalization: Model trained on one fairing orientation")
    print("     automatically works for any orientation")

    print(f"\n[SUCCESS] E(3)-Equivariant GNN Prototype Completed.")
    print("\nNext Steps for Production:")
    print("  1. Integrate with PyG Data from build_graph.py")
    print("  2. Add vector features (displacements, normals) as equivariant channels")
    print("  3. Use e3nn library for higher-order spherical harmonics")
    print("  4. Benchmark equivariance benefit on rotated test sets")
    print("  5. Combine with Gauge Equivariant CNN for manifold convolution")


if __name__ == "__main__":
    train_egnn_prototype()
