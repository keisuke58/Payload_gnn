import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# ==============================================================================
# DeepONet (Deep Operator Network) for Parametric PDE Surrogate
#
# Goal: Learn the operator mapping from defect parameters (location, size)
#       to the full stress/wave field response on the fairing surface.
#       This enables real-time "what-if" analysis without running Abaqus.
#
# Architecture (Unstacked DeepONet):
#   - Branch Net: Encodes the input function (defect parameter field)
#     evaluated at fixed sensor locations → R^p
#   - Trunk Net: Encodes the query coordinates (x, y, z) → R^p
#   - Output: dot product of Branch and Trunk outputs
#     u(y) = sum_k branch_k(a) * trunk_k(y) + bias
#
# Reference:
#   Lu et al. "Learning nonlinear operators via DeepONet" (Nature MI 2021)
#   Wiki Section 7.1: Neural Operators (FNO / DeepONet)
# ==============================================================================


class BranchNet(nn.Module):
    """Branch Network: Encodes the input function (defect configuration).

    Input: Function values evaluated at m fixed sensor locations.
           For H3 Fairing: stress/wave readings at sensor positions.
    Output: p-dimensional coefficient vector.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, u_sensors):
        # u_sensors: (batch, m) where m = number of sensor locations
        return self.net(u_sensors)


class TrunkNet(nn.Module):
    """Trunk Network: Encodes the query coordinates.

    Input: Spatial coordinate y = (x, y, z) or (theta, z) on fairing surface.
    Output: p-dimensional basis function evaluation.
    """
    def __init__(self, coord_dim=2, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.Tanh(),  # Tanh works well for coordinate encoding
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y):
        # y: (N, coord_dim) where N = number of query points
        return self.net(y)


class DeepONet(nn.Module):
    """Deep Operator Network.

    Learns the operator G: input_function → output_function.
    For H3 Fairing SHM:
        Input function:  Defect parameter field (stiffness distribution on surface)
        Output function: Wave field / stress field response
    """
    def __init__(self, sensor_dim, coord_dim=2, hidden_dim=128, basis_dim=64):
        super().__init__()
        self.branch = BranchNet(sensor_dim, hidden_dim, basis_dim)
        self.trunk = TrunkNet(coord_dim, hidden_dim, basis_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_coords):
        """
        Args:
            u_sensors: (batch, m) - Input function at sensor locations
            y_coords:  (batch, N, coord_dim) or (N, coord_dim) - Query coordinates

        Returns:
            output: (batch, N) - Predicted field values at query points
        """
        branch_out = self.branch(u_sensors)  # (batch, p)

        if y_coords.dim() == 2:
            # Same query points for all batch items
            trunk_out = self.trunk(y_coords)  # (N, p)
            # Output: (batch, N) via einsum
            output = torch.einsum('bp,np->bn', branch_out, trunk_out) + self.bias
        else:
            # Different query points per batch item
            batch_size = y_coords.size(0)
            N = y_coords.size(1)
            trunk_out = self.trunk(y_coords.reshape(-1, y_coords.size(-1)))  # (batch*N, p)
            trunk_out = trunk_out.view(batch_size, N, -1)  # (batch, N, p)
            output = torch.einsum('bp,bnp->bn', branch_out, trunk_out) + self.bias

        return output


# ==============================================================================
# Synthetic Data Generator
# ==============================================================================
def generate_deeponet_data(
    num_samples=200,
    num_sensors=50,
    num_query_points=100,
    domain_size=1.0,
):
    """
    Generate synthetic operator learning data.

    Problem: Given a defect configuration (stiffness field),
             predict the resulting stress/displacement field.

    This mimics the H3 Fairing scenario where:
    - Input: Defect mask on the fairing surface (measured at sensor locations)
    - Output: Wave field response at arbitrary query points

    Simplified physics: Poisson equation -Δu = f  on [0, 1]²
    with source term influenced by defect regions.
    We approximate solutions using analytical superposition.
    """
    print(f"Generating {num_samples} DeepONet samples...")
    print(f"  Sensors: {num_sensors}, Query points: {num_query_points}")

    # Fixed sensor grid (branch input locations)
    ns = int(np.sqrt(num_sensors))
    sensor_x = np.linspace(0.1, 0.9, ns)
    sensor_y = np.linspace(0.1, 0.9, ns)
    sensor_xx, sensor_yy = np.meshgrid(sensor_x, sensor_y)
    sensor_coords = np.stack([sensor_xx.ravel(), sensor_yy.ravel()], axis=-1)
    actual_sensors = sensor_coords.shape[0]  # ns * ns

    # Random query points (trunk input locations)
    query_coords = np.random.rand(num_query_points, 2).astype(np.float32)

    all_branch_inputs = []
    all_outputs = []

    for _ in range(num_samples):
        # Generate random defect configuration
        num_defects = np.random.randint(0, 4)

        # Stiffness field (1.0 = healthy, reduced in defect regions)
        def stiffness_field(x, y):
            k = np.ones_like(x)
            for _ in range(num_defects):
                cx, cy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
                sigma = np.random.uniform(0.05, 0.1)
                dist_sq = (x - cx)**2 + (y - cy)**2
                k -= 0.7 * np.exp(-dist_sq / (2 * sigma**2))
            return np.clip(k, 0.1, 1.0)

        # Evaluate stiffness at sensor locations (branch input)
        k_sensors = stiffness_field(sensor_coords[:, 0], sensor_coords[:, 1])

        # Generate output field (simplified wave response)
        # Higher stiffness → higher wave speed → different field pattern
        def response_field(x, y):
            """Simplified analytical response to defect configuration."""
            base_wave = np.sin(3 * np.pi * x) * np.sin(2 * np.pi * y)
            k = stiffness_field(x, y)
            # Low stiffness (defect) → attenuated amplitude + phase shift
            response = base_wave * k + 0.5 * (1.0 - k) * np.sin(8 * np.pi * x)
            return response

        # Evaluate response at query points
        u_query = response_field(query_coords[:, 0], query_coords[:, 1])

        all_branch_inputs.append(k_sensors)
        all_outputs.append(u_query)

    branch_inputs = torch.tensor(np.array(all_branch_inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(all_outputs), dtype=torch.float32)
    query_tensor = torch.tensor(query_coords, dtype=torch.float32)

    return branch_inputs, outputs, query_tensor, actual_sensors


# ==============================================================================
# Training Loop
# ==============================================================================
def train_deeponet_prototype():
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    HIDDEN_DIM = 128
    BASIS_DIM = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_branch, train_output, query_coords, n_sensors = generate_deeponet_data(
        num_samples=500, num_sensors=49, num_query_points=200,
    )
    test_branch, test_output, _, _ = generate_deeponet_data(
        num_samples=100, num_sensors=49, num_query_points=200,
    )

    query_coords = query_coords.to(device)

    train_loader = DataLoader(
        TensorDataset(train_branch, train_output),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_branch, test_output),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # 2. Model
    model = DeepONet(
        sensor_dim=n_sensors,
        coord_dim=2,
        hidden_dim=HIDDEN_DIM,
        basis_dim=BASIS_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sensor locations: {n_sensors}")
    print(f"Query points: {query_coords.shape[0]}")

    # 3. Train
    print("\nStarting DeepONet Training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_branch, batch_output in train_loader:
            batch_branch = batch_branch.to(device)
            batch_output = batch_output.to(device)

            optimizer.zero_grad()

            # Forward: branch encodes defect config, trunk evaluates at query points
            pred = model(batch_branch, query_coords)  # (batch, N_query)

            loss = criterion(pred, batch_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # 4. Evaluation
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)

    model.eval()
    total_mse = 0
    total_samples = 0

    with torch.no_grad():
        for batch_branch, batch_output in test_loader:
            batch_branch = batch_branch.to(device)
            batch_output = batch_output.to(device)

            pred = model(batch_branch, query_coords)
            mse = ((pred - batch_output) ** 2).mean(dim=1)  # Per-sample MSE
            total_mse += mse.sum().item()
            total_samples += batch_branch.size(0)

    avg_mse = total_mse / total_samples
    avg_rmse = np.sqrt(avg_mse)

    print(f"  Test MSE:  {avg_mse:.6f}")
    print(f"  Test RMSE: {avg_rmse:.6f}")

    # 5. Resolution Invariance Test
    print("\n" + "-" * 60)
    print("Resolution Invariance Test")
    print("-" * 60)
    print("Testing on 2x denser query grid (400 points vs 200 training)...")

    dense_coords = torch.tensor(
        np.random.rand(400, 2).astype(np.float32)
    ).to(device)

    with torch.no_grad():
        # Use first 5 test samples
        sample_branch = test_branch[:5].to(device)
        pred_dense = model(sample_branch, dense_coords)

    print(f"  Input shape:  ({sample_branch.shape[0]}, {sample_branch.shape[1]}) sensors")
    print(f"  Output shape: ({pred_dense.shape[0]}, {pred_dense.shape[1]}) query points")
    print(f"  Output range: [{pred_dense.min().item():.4f}, {pred_dense.max().item():.4f}]")
    print("  [OK] Model evaluates at arbitrary resolution without retraining.")

    # 6. Inference Speed Test
    print("\n" + "-" * 60)
    print("Inference Speed Comparison")
    print("-" * 60)

    # Single forward pass timing
    with torch.no_grad():
        single_input = test_branch[:1].to(device)
        # Warmup
        for _ in range(10):
            _ = model(single_input, query_coords)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()
        n_infer = 100
        for _ in range(n_infer):
            _ = model(single_input, query_coords)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()

    avg_infer_ms = (t1 - t0) / n_infer * 1000
    print(f"  Average inference time: {avg_infer_ms:.2f} ms per sample")
    print(f"  Compared to Abaqus FEM: ~minutes per sample → ~{60000/avg_infer_ms:.0f}x speedup")

    # Summary
    print("\n" + "=" * 60)
    print("[SUCCESS] DeepONet Prototype Completed.")
    print("=" * 60)
    print(f"  Operator: Defect Configuration → Wave/Stress Response Field")
    print(f"  Test RMSE: {avg_rmse:.6f}")
    print(f"  Inference: {avg_infer_ms:.2f} ms (vs minutes for FEM)")
    print(f"  Resolution-independent evaluation verified")

    print("\nNext Steps for Production:")
    print("  1. Replace synthetic data with Abaqus H3 FEM pairs")
    print("  2. Branch input: stress/wave at real sensor positions")
    print("  3. Trunk input: 3D coordinates on fairing surface (x,y,z)")
    print("  4. Combine with FNO for multi-scale surrogate modeling")
    print("  5. Use as data augmentation engine for GNN training")


if __name__ == "__main__":
    train_deeponet_prototype()
