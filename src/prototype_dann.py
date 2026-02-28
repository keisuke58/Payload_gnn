import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# ==============================================================================
# DANN (Domain Adversarial Neural Network) for Sim-to-Real Transfer
#
# Goal: Train a defect detector on Abaqus simulation data (Source Domain)
#       and adapt it to work on real experimental data (Target Domain)
#       using adversarial domain alignment.
#
# Architecture:
#   - Feature Extractor (shared): Encodes node features from both domains
#   - Label Predictor: Classifies defect vs. healthy (trained on source labels)
#   - Domain Discriminator: Distinguishes Sim vs. Real (with Gradient Reversal)
#
# Reference:
#   Ganin et al. "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
#   Phase 4 in ML_STRATEGY_AND_IMPLEMENTATION.md
# ==============================================================================


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL).

    During forward pass: identity function.
    During backward pass: negates the gradient by -lambda.
    This forces the feature extractor to learn domain-invariant features.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class FeatureExtractor(nn.Module):
    """Shared feature extractor for both simulation and real domains.

    In production, this would be replaced by one of:
    - GCN/GAT/GIN from src/models.py (graph-based)
    - PointTransformer from src/models_point.py (point cloud)

    For this prototype, we use an MLP on tabular node features.
    """
    def __init__(self, in_features, hidden_dim=128, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    """Defect classifier head.
    Binary: 0 = Healthy, 1 = Defect.
    """
    def __init__(self, feature_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class DomainDiscriminator(nn.Module):
    """Domain classifier head with Gradient Reversal Layer.
    Binary: 0 = Simulation (Abaqus), 1 = Real (Experiment / OGW).
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, features):
        reversed_features = self.grl(features)
        return self.net(reversed_features)


class DANN(nn.Module):
    """Domain Adversarial Neural Network.

    Combines feature extractor, label predictor, and domain discriminator
    into a single end-to-end model.
    """
    def __init__(self, in_features, hidden_dim=128, feature_dim=64, num_classes=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_features, hidden_dim, feature_dim)
        self.label_predictor = LabelPredictor(feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(feature_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_discriminator(features)
        return class_output, domain_output

    def set_lambda(self, lambda_):
        """Update GRL lambda (schedule during training)."""
        self.domain_discriminator.grl.set_lambda(lambda_)


# ==============================================================================
# Synthetic Data Generator
# ==============================================================================
def generate_synthetic_domains(
    n_source=500,
    n_target=500,
    n_features=10,
    defect_ratio=0.2,
    domain_shift=0.5,
):
    """
    Generate synthetic source (simulation) and target (real) domain data.

    Source Domain (Abaqus Simulation):
        - Clean, physics-consistent features
        - Labeled (defect / healthy)

    Target Domain (Real Experiment / OGW):
        - Features shifted by damping, noise, sensor coupling effects
        - Labels available only for evaluation (not used in training)

    Feature vector (mimics H3 fairing node features):
        [stress_s11, stress_s22, stress_s12, displacement, temperature,
         thermal_stress, curvature_k1, curvature_k2, normal_z, wave_amplitude]
    """
    print(f"Generating synthetic domain data...")
    print(f"  Source (Sim): {n_source} nodes, Target (Real): {n_target} nodes")

    # --- Source Domain (Simulation) ---
    # Base healthy features: centered around 0, unit variance
    X_source = np.random.randn(n_source, n_features).astype(np.float32)
    y_source = np.zeros(n_source, dtype=np.int64)

    # Inject defects: shift stress features + add wave scattering
    n_defect = int(n_source * defect_ratio)
    defect_idx = np.random.choice(n_source, n_defect, replace=False)
    X_source[defect_idx, 0:3] += 2.0   # Elevated stresses (s11, s22, s12)
    X_source[defect_idx, 3] += 1.5     # Larger displacement
    X_source[defect_idx, 9] -= 1.0     # Attenuated wave amplitude
    y_source[defect_idx] = 1

    # --- Target Domain (Real Experiment) ---
    # Same structure but with domain shift
    X_target = np.random.randn(n_target, n_features).astype(np.float32)
    y_target = np.zeros(n_target, dtype=np.int64)

    # Global domain shift: real data has different noise profile, damping
    X_target += domain_shift                     # Mean shift (sensor calibration)
    X_target *= (1.0 + 0.3 * np.random.randn())  # Scale shift (material variability)
    X_target += 0.2 * np.random.randn(n_target, n_features).astype(np.float32)  # Extra noise

    # Inject defects (same pattern but through real-world distortion)
    n_defect_target = int(n_target * defect_ratio)
    defect_idx_target = np.random.choice(n_target, n_defect_target, replace=False)
    X_target[defect_idx_target, 0:3] += 1.8     # Slightly different stress response
    X_target[defect_idx_target, 3] += 1.2        # Different displacement sensitivity
    X_target[defect_idx_target, 9] -= 0.8        # Different wave attenuation
    y_target[defect_idx_target] = 1

    # Convert to tensors
    X_source = torch.from_numpy(X_source)
    y_source = torch.from_numpy(y_source)
    X_target = torch.from_numpy(X_target)
    y_target = torch.from_numpy(y_target)

    # Domain labels: 0 = Source (Sim), 1 = Target (Real)
    d_source = torch.zeros(n_source, dtype=torch.long)
    d_target = torch.ones(n_target, dtype=torch.long)

    print(f"  Source defects: {n_defect}/{n_source} ({defect_ratio*100:.0f}%)")
    print(f"  Target defects: {n_defect_target}/{n_target} ({defect_ratio*100:.0f}%)")
    print(f"  Domain shift magnitude: {domain_shift}")

    return X_source, y_source, d_source, X_target, y_target, d_target


def compute_grl_lambda(epoch, max_epoch):
    """Progressive GRL lambda schedule (Ganin et al. 2016).

    Starts from 0, gradually increases to 1 following a sigmoid schedule.
    This prevents the domain adversarial signal from destabilizing early training.
    """
    p = epoch / max_epoch
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


# ==============================================================================
# Training Loop
# ==============================================================================
def train_dann_prototype():
    # Settings
    N_FEATURES = 10
    HIDDEN_DIM = 128
    FEATURE_DIM = 64
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Generate Data
    X_src, y_src, d_src, X_tgt, y_tgt, d_tgt = generate_synthetic_domains(
        n_source=1000, n_target=1000, n_features=N_FEATURES,
        defect_ratio=0.2, domain_shift=0.5,
    )

    src_loader = DataLoader(
        TensorDataset(X_src, y_src, d_src),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    tgt_loader = DataLoader(
        TensorDataset(X_tgt, y_tgt, d_tgt),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    # 2. Model
    model = DANN(N_FEATURES, HIDDEN_DIM, FEATURE_DIM, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    # 3. Train
    print("\nStarting DANN Training (Source: Sim, Target: Real)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        # Update GRL lambda (progressive schedule)
        grl_lambda = compute_grl_lambda(epoch, EPOCHS)
        model.set_lambda(grl_lambda)

        total_label_loss = 0
        total_domain_loss = 0
        n_batches = 0

        tgt_iter = iter(tgt_loader)

        for src_batch in src_loader:
            x_s, y_s, d_s = [t.to(device) for t in src_batch]

            # Get target batch (cycle if shorter)
            try:
                x_t, _, d_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_t, _, d_t = next(tgt_iter)
            x_t, d_t = x_t.to(device), d_t.to(device)

            optimizer.zero_grad()

            # --- Source forward ---
            class_out_s, domain_out_s = model(x_s)
            loss_label = label_criterion(class_out_s, y_s)
            loss_domain_s = domain_criterion(domain_out_s, d_s)

            # --- Target forward (no label loss, only domain) ---
            _, domain_out_t = model(x_t)
            loss_domain_t = domain_criterion(domain_out_t, d_t)

            # Total loss
            loss_domain = (loss_domain_s + loss_domain_t) / 2.0
            loss = loss_label + loss_domain

            loss.backward()
            optimizer.step()

            total_label_loss += loss_label.item()
            total_domain_loss += loss_domain.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_label = total_label_loss / n_batches
            avg_domain = total_domain_loss / n_batches
            print(f"Epoch {epoch+1}/{EPOCHS} | Label Loss: {avg_label:.4f} | "
                  f"Domain Loss: {avg_domain:.4f} | GRL λ: {grl_lambda:.3f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # 4. Evaluation
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    model.eval()

    # --- Source domain (should be good: trained on these labels) ---
    with torch.no_grad():
        class_out_s, _ = model(X_src.to(device))
        pred_s = class_out_s.argmax(dim=1).cpu()
        acc_s = (pred_s == y_src).float().mean().item()

        # Defect metrics
        tp_s = ((pred_s == 1) & (y_src == 1)).sum().item()
        fn_s = ((pred_s == 0) & (y_src == 1)).sum().item()
        fp_s = ((pred_s == 1) & (y_src == 0)).sum().item()
        recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
        precision_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
        f1_s = 2 * precision_s * recall_s / (precision_s + recall_s) if (precision_s + recall_s) > 0 else 0

    print(f"\n[Source Domain - Simulation]")
    print(f"  Accuracy:  {acc_s:.4f}")
    print(f"  Precision: {precision_s:.4f}")
    print(f"  Recall:    {recall_s:.4f}")
    print(f"  F1 Score:  {f1_s:.4f}")

    # --- Target domain (key metric: can we generalize?) ---
    with torch.no_grad():
        class_out_t, _ = model(X_tgt.to(device))
        pred_t = class_out_t.argmax(dim=1).cpu()
        acc_t = (pred_t == y_tgt).float().mean().item()

        tp_t = ((pred_t == 1) & (y_tgt == 1)).sum().item()
        fn_t = ((pred_t == 0) & (y_tgt == 1)).sum().item()
        fp_t = ((pred_t == 1) & (y_tgt == 0)).sum().item()
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0

    print(f"\n[Target Domain - Real (Domain Adapted)]")
    print(f"  Accuracy:  {acc_t:.4f}")
    print(f"  Precision: {precision_t:.4f}")
    print(f"  Recall:    {recall_t:.4f}")
    print(f"  F1 Score:  {f1_t:.4f}")

    # --- Baseline: Source-only model (no DANN) for comparison ---
    print("\n" + "-" * 60)
    print("Training Source-Only Baseline (No Domain Adaptation)...")

    baseline_model = nn.Sequential(
        FeatureExtractor(N_FEATURES, HIDDEN_DIM, FEATURE_DIM),
        LabelPredictor(FEATURE_DIM, 2),
    ).to(device)

    baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        baseline_model.train()
        for x_s, y_s, _ in src_loader:
            x_s, y_s = x_s.to(device), y_s.to(device)
            baseline_opt.zero_grad()
            out = baseline_model(x_s)
            loss = label_criterion(out, y_s)
            loss.backward()
            baseline_opt.step()

    baseline_model.eval()
    with torch.no_grad():
        out_t_baseline = baseline_model(X_tgt.to(device))
        pred_t_baseline = out_t_baseline.argmax(dim=1).cpu()
        acc_t_baseline = (pred_t_baseline == y_tgt).float().mean().item()

        tp_b = ((pred_t_baseline == 1) & (y_tgt == 1)).sum().item()
        fn_b = ((pred_t_baseline == 0) & (y_tgt == 1)).sum().item()
        fp_b = ((pred_t_baseline == 1) & (y_tgt == 0)).sum().item()
        recall_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        precision_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        f1_b = 2 * precision_b * recall_b / (precision_b + recall_b) if (precision_b + recall_b) > 0 else 0

    print(f"\n[Baseline: Source-Only → Target Domain]")
    print(f"  Accuracy:  {acc_t_baseline:.4f}")
    print(f"  Precision: {precision_b:.4f}")
    print(f"  Recall:    {recall_b:.4f}")
    print(f"  F1 Score:  {f1_b:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary: DANN vs Source-Only on Target Domain")
    print("=" * 60)
    print(f"  DANN F1:         {f1_t:.4f}")
    print(f"  Source-Only F1:  {f1_b:.4f}")
    improvement = f1_t - f1_b
    if improvement > 0:
        print(f"  Improvement:     +{improvement:.4f}")
        print(f"\n[SUCCESS] DANN improved target domain performance!")
    else:
        print(f"  Difference:      {improvement:.4f}")
        print(f"\n[NOTE] Synthetic data may not show large gap. "
              "Real Abaqus→OGW transfer will have stronger domain shift.")

    print("\nNext Steps for Production:")
    print("  1. Replace FeatureExtractor with GNN from src/models.py")
    print("  2. Use Abaqus H3 dataset as Source domain")
    print("  3. Use Open Guided Waves (OGW) data as Target domain")
    print("  4. Integrate with src/train.py training infrastructure")


if __name__ == "__main__":
    train_dann_prototype()
