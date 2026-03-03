# -*- coding: utf-8 -*-
"""
Uncertainty Quantification for GNN Defect Localization

Two complementary methods:
  1. MC Dropout — single model, T stochastic forward passes
  2. Deep Ensemble — K fold models, variance across predictions

Both produce per-node:
  - mean P(defect)
  - epistemic uncertainty (std of P(defect))
  - predictive entropy

Usage:
    from uncertainty import mc_dropout_predict, ensemble_predict_with_uncertainty

    # MC Dropout (single model)
    mean_prob, std_prob, entropy = mc_dropout_predict(model, data, T=30)

    # Deep Ensemble (5-fold)
    mean_prob, std_prob, entropy = ensemble_predict_with_uncertainty(models, data)
"""

import numpy as np
import torch
import torch.nn.functional as F


# =========================================================================
# MC Dropout
# =========================================================================
@torch.no_grad()
def mc_dropout_predict(model, x, edge_index, edge_attr, batch=None, T=30):
    """Run T stochastic forward passes with dropout enabled.

    Args:
        model: trained GNN model (with dropout layers)
        x, edge_index, edge_attr, batch: graph data tensors
        T: number of MC samples

    Returns:
        mean_prob: (N,) mean P(defect) across T passes
        std_prob: (N,) std P(defect) — epistemic uncertainty
        entropy: (N,) predictive entropy
    """
    model.train()  # enable dropout
    probs_list = []

    for _ in range(T):
        logits = model(x, edge_index, edge_attr, batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        probs_list.append(probs.cpu())

    model.eval()

    probs_stack = torch.stack(probs_list, dim=0)  # (T, N)
    mean_prob = probs_stack.mean(dim=0)  # (N,)
    std_prob = probs_stack.std(dim=0)    # (N,)

    # Predictive entropy: H = -Σ p*log(p)
    p1 = mean_prob.clamp(1e-8, 1 - 1e-8)
    p0 = 1 - p1
    entropy = -(p1 * p1.log() + p0 * p0.log())

    return mean_prob.numpy(), std_prob.numpy(), entropy.numpy()


# =========================================================================
# Deep Ensemble
# =========================================================================
@torch.no_grad()
def ensemble_predict_with_uncertainty(models, x, edge_index, edge_attr,
                                      batch=None):
    """Run inference across K ensemble models and compute uncertainty.

    Args:
        models: list of trained GNN models
        x, edge_index, edge_attr, batch: graph data tensors

    Returns:
        mean_prob: (N,) mean P(defect)
        std_prob: (N,) epistemic uncertainty
        entropy: (N,) predictive entropy
    """
    probs_list = []
    for model in models:
        model.eval()
        logits = model(x, edge_index, edge_attr, batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        probs_list.append(probs.cpu())

    probs_stack = torch.stack(probs_list, dim=0)  # (K, N)
    mean_prob = probs_stack.mean(dim=0)
    std_prob = probs_stack.std(dim=0)

    p1 = mean_prob.clamp(1e-8, 1 - 1e-8)
    p0 = 1 - p1
    entropy = -(p1 * p1.log() + p0 * p0.log())

    return mean_prob.numpy(), std_prob.numpy(), entropy.numpy()


# =========================================================================
# Combined: MC Dropout × Deep Ensemble
# =========================================================================
@torch.no_grad()
def ensemble_mc_predict(models, x, edge_index, edge_attr, batch=None, T=10):
    """Combined uncertainty: MC Dropout within each ensemble member.

    Total samples = K * T. Decomposes uncertainty into:
      - epistemic (model uncertainty): std across ensemble means
      - aleatoric approx: mean of within-model stds

    Returns:
        mean_prob, total_std, epistemic_std, entropy
    """
    all_probs = []  # K*T samples
    model_means = []

    for model in models:
        model.train()
        member_probs = []
        for _ in range(T):
            logits = model(x, edge_index, edge_attr, batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu()
            member_probs.append(probs)
            all_probs.append(probs)
        model.eval()
        model_means.append(torch.stack(member_probs).mean(dim=0))

    all_stack = torch.stack(all_probs, dim=0)  # (K*T, N)
    mean_prob = all_stack.mean(dim=0)
    total_std = all_stack.std(dim=0)

    model_means_stack = torch.stack(model_means, dim=0)  # (K, N)
    epistemic_std = model_means_stack.std(dim=0)

    p1 = mean_prob.clamp(1e-8, 1 - 1e-8)
    p0 = 1 - p1
    entropy = -(p1 * p1.log() + p0 * p0.log())

    return (mean_prob.numpy(), total_std.numpy(),
            epistemic_std.numpy(), entropy.numpy())


# =========================================================================
# Calibration metrics
# =========================================================================
def expected_calibration_error(probs, targets, n_bins=10):
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better.

    Args:
        probs: (N,) predicted P(defect)
        targets: (N,) binary ground truth
        n_bins: number of calibration bins

    Returns:
        ece: scalar ECE value
        bin_accs: per-bin accuracy
        bin_confs: per-bin mean confidence
        bin_counts: per-bin sample count
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        count = mask.sum()
        if count > 0:
            bin_accs[i] = targets[mask].mean()
            bin_confs[i] = probs[mask].mean()
            bin_counts[i] = count

    total = bin_counts.sum()
    if total == 0:
        return 0.0, bin_accs, bin_confs, bin_counts

    ece = (bin_counts / total * np.abs(bin_accs - bin_confs)).sum()
    return float(ece), bin_accs, bin_confs, bin_counts


def uncertainty_quality_metrics(probs, uncertainty, targets):
    """Compute metrics evaluating uncertainty quality.

    Args:
        probs: (N,) mean P(defect)
        uncertainty: (N,) epistemic uncertainty (std)
        targets: (N,) binary ground truth

    Returns:
        dict with uncertainty quality metrics
    """
    preds = (probs >= 0.5).astype(int)
    correct = (preds == targets)
    incorrect = ~correct

    metrics = {}

    # Mean uncertainty for correct vs incorrect predictions
    if correct.sum() > 0:
        metrics['uncertainty_correct_mean'] = float(uncertainty[correct].mean())
    if incorrect.sum() > 0:
        metrics['uncertainty_incorrect_mean'] = float(uncertainty[incorrect].mean())

    # Uncertainty should be higher for incorrect predictions
    if correct.sum() > 0 and incorrect.sum() > 0:
        metrics['uncertainty_separation'] = float(
            uncertainty[incorrect].mean() - uncertainty[correct].mean()
        )

    # AUROC of uncertainty as error detector
    # High uncertainty → likely error → uncertainty as binary classifier
    if correct.sum() > 0 and incorrect.sum() > 0:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['uncertainty_auroc'] = float(
                roc_auc_score(incorrect.astype(int), uncertainty)
            )
        except ValueError:
            metrics['uncertainty_auroc'] = 0.0

    # ECE
    ece, _, _, _ = expected_calibration_error(probs, targets)
    metrics['ece'] = ece

    # Coverage at uncertainty threshold
    # "Reject" predictions with high uncertainty
    for reject_pct in (5, 10, 20):
        threshold = np.percentile(uncertainty, 100 - reject_pct)
        keep_mask = uncertainty <= threshold
        if keep_mask.sum() > 0:
            kept_acc = (preds[keep_mask] == targets[keep_mask]).mean()
            metrics['accuracy_reject_%d%%' % reject_pct] = float(kept_acc)

    return metrics


# =========================================================================
# Visualization
# =========================================================================
def plot_uncertainty_map(pos, probs, uncertainty, targets, save_path):
    """Plot 3D uncertainty heatmap alongside prediction and ground truth.

    Four panels: Ground Truth | Prediction | Uncertainty | Error+Uncertainty
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping uncertainty visualization.")
        return

    fig = plt.figure(figsize=(24, 6))

    # 1. Ground truth
    ax1 = fig.add_subplot(141, projection='3d')
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=targets, cmap='RdYlBu_r', s=1, vmin=0, vmax=1)
    ax1.set_title('Ground Truth')
    plt.colorbar(sc1, ax=ax1, shrink=0.5)

    # 2. Mean prediction
    ax2 = fig.add_subplot(142, projection='3d')
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=probs, cmap='RdYlBu_r', s=1, vmin=0, vmax=1)
    ax2.set_title('P(defect) — Mean')
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    # 3. Epistemic uncertainty
    ax3 = fig.add_subplot(143, projection='3d')
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=uncertainty, cmap='hot', s=1, vmin=0)
    ax3.set_title('Uncertainty (std)')
    plt.colorbar(sc3, ax=ax3, shrink=0.5)

    # 4. Error overlay: red=FP, blue=FN, size proportional to uncertainty
    preds = (probs >= 0.5).astype(int)
    error_map = np.zeros(len(pos))
    error_map[(preds == 1) & (targets == 0)] = 1.0   # FP
    error_map[(preds == 0) & (targets == 1)] = -1.0   # FN

    sizes = 1 + uncertainty * 50  # scale uncertainty to point size

    ax4 = fig.add_subplot(144, projection='3d')
    sc4 = ax4.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=error_map, cmap='coolwarm', s=sizes, vmin=-1, vmax=1,
                      alpha=0.7)
    ax4.set_title('Error (size=uncertainty)')
    plt.colorbar(sc4, ax=ax4, shrink=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved uncertainty map: %s" % save_path)


def plot_calibration_curve(probs, targets, save_path, n_bins=10):
    """Plot reliability diagram (calibration curve)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        probs, targets, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    valid = bin_counts > 0
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(bin_confs[valid], bin_accs[valid],
            width=1.0 / n_bins, alpha=0.6, edgecolor='black',
            align='center', label='Model')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.hist(probs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('P(defect)')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved calibration curve: %s" % save_path)
