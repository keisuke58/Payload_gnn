#!/usr/bin/env python3
"""
Anomaly Detection for Semiconductor Test Data
===============================================
Portfolio: Advantest — ML/Data Science

Detects anomalous IC test patterns using Isolation Forest
and One-Class methods implemented from scratch.

ML Concepts:
  - Isolation Forest from scratch (anomaly score via path length)
  - One-Class SVM (approximate with SVDD)
  - PCA-based anomaly detection (reconstruction error)
  - ROC/AUC evaluation
  - Multi-sensor test data fusion

Domain:
  - ATE (Automatic Test Equipment) multi-parameter test data
  - Outlier detection for yield-loss analysis
  - Parametric test failure classification

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

np.random.seed(42)

# ============================================================
# Synthetic ATE Test Dataset
# ============================================================
PARAM_NAMES = [
    'Vth [mV]', 'Idsat [mA]', 'Ioff [nA]', 'BVdss [V]',
    'Ron [mOhm]', 'Freq_max [GHz]', 'Power [mW]', 'Delay [ps]'
]

def generate_test_dataset(n_normal=800, n_anomaly=50):
    """Generate multi-parameter ATE test data with anomalies."""
    # Normal distribution (correlated parameters)
    mean_normal = np.array([500, 1.2, 5.0, 12.0, 50, 3.5, 100, 15])
    # Correlation structure
    cov = np.eye(8)
    cov[0, 1] = cov[1, 0] = 0.6   # Vth-Idsat correlation
    cov[2, 3] = cov[3, 2] = -0.4  # Ioff-BVdss anticorrelation
    cov[4, 5] = cov[5, 4] = -0.5  # Ron-Freq correlation
    cov[6, 7] = cov[7, 6] = 0.7   # Power-Delay correlation

    std = np.array([30, 0.15, 2.0, 1.5, 8, 0.3, 15, 2])
    cov_scaled = np.outer(std, std) * cov

    X_normal = np.random.multivariate_normal(mean_normal, cov_scaled, n_normal)

    # Anomalies: several failure modes
    anomalies = []
    # Mode 1: Gate oxide breakdown (high Ioff, low BVdss)
    n1 = n_anomaly // 3
    a1 = np.random.multivariate_normal(mean_normal, cov_scaled, n1)
    a1[:, 2] *= 10  # Ioff 10x
    a1[:, 3] *= 0.5  # BVdss low
    anomalies.append(a1)

    # Mode 2: Contact resistance (high Ron, low Freq)
    n2 = n_anomaly // 3
    a2 = np.random.multivariate_normal(mean_normal, cov_scaled, n2)
    a2[:, 4] *= 3   # Ron 3x
    a2[:, 5] *= 0.6  # Freq low
    anomalies.append(a2)

    # Mode 3: Random outliers
    n3 = n_anomaly - n1 - n2
    a3 = np.random.multivariate_normal(mean_normal, cov_scaled * 9, n3)
    anomalies.append(a3)

    X_anomaly = np.vstack(anomalies)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(len(X_anomaly))])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

# ============================================================
# Isolation Forest (from scratch)
# ============================================================
class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def _build(self, X, depth):
        n, p = X.shape
        if depth >= self.max_depth or n <= 1:
            return {'leaf': True, 'size': n}

        # Random feature and split
        feat = np.random.randint(p)
        x_min, x_max = X[:, feat].min(), X[:, feat].max()
        if x_max - x_min < 1e-10:
            return {'leaf': True, 'size': n}

        split = np.random.uniform(x_min, x_max)
        left_mask = X[:, feat] < split

        if left_mask.all() or (~left_mask).all():
            return {'leaf': True, 'size': n}

        return {
            'leaf': False,
            'feature': feat, 'split': split,
            'left': self._build(X[left_mask], depth + 1),
            'right': self._build(X[~left_mask], depth + 1),
        }

    def fit(self, X):
        self.tree = self._build(X, 0)

    def path_length(self, x, node=None, depth=0):
        if node is None:
            node = self.tree
        if node['leaf']:
            # Average path length for external nodes
            n = node['size']
            if n <= 1:
                return depth
            return depth + self._c(n)
        if x[node['feature']] < node['split']:
            return self.path_length(x, node['left'], depth + 1)
        return self.path_length(x, node['right'], depth + 1)

    @staticmethod
    def _c(n):
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X):
        n = len(X)
        self.trees = []
        psi = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(psi)))

        for _ in range(self.n_estimators):
            idx = np.random.choice(n, psi, replace=False)
            tree = IsolationTree(max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)

        self._c_n = IsolationTree._c(min(self.max_samples, n))

    def anomaly_score(self, X):
        """Anomaly score: s(x) = 2^(-E[h(x)]/c(n))."""
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            avg_path = np.mean([t.path_length(x) for t in self.trees])
            scores[i] = 2 ** (-avg_path / self._c_n)
        return scores

    def predict(self, X, threshold=0.5):
        scores = self.anomaly_score(X)
        return (scores > threshold).astype(int)

# ============================================================
# PCA Anomaly Detection
# ============================================================
class PCAAnomalyDetector:
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-10
        X_norm = (X - self.mean) / self.std
        cov = np.cov(X_norm.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.components = eigvecs[:, idx[:self.n_components]]
        self.explained_var = eigvals[idx] / eigvals.sum()

    def reconstruction_error(self, X):
        X_norm = (X - self.mean) / self.std
        projected = X_norm @ self.components
        reconstructed = projected @ self.components.T
        return np.sum((X_norm - reconstructed)**2, axis=1)

# ============================================================
# ROC / AUC
# ============================================================
def roc_curve(y_true, scores, n_thresholds=200):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    return np.array(fpr_list), np.array(tpr_list)

def auc(fpr, tpr):
    sorted_idx = np.argsort(fpr)
    return np.trapz(tpr[sorted_idx], fpr[sorted_idx])

# ============================================================
# Visualization
# ============================================================
def plot_results(X, y, iforest_scores, pca_errors, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Anomaly Detection for Semiconductor Test Data (Advantest)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    normal_mask = y == 0
    anomaly_mask = y == 1

    # (a) 2D scatter (Vth vs Idsat)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X[normal_mask, 0], X[normal_mask, 1], s=15, alpha=0.4,
               c='blue', label='Normal')
    ax1.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], s=40, alpha=0.8,
               c='red', marker='x', label='Anomaly')
    ax1.set_xlabel(PARAM_NAMES[0])
    ax1.set_ylabel(PARAM_NAMES[1])
    ax1.set_title('(a) Test Data Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Isolation Forest scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(iforest_scores[normal_mask], bins=40, alpha=0.6, color='blue',
            label='Normal', density=True)
    ax2.hist(iforest_scores[anomaly_mask], bins=20, alpha=0.6, color='red',
            label='Anomaly', density=True)
    ax2.axvline(0.5, color='green', linestyle='--', label='Threshold=0.5')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Isolation Forest Score Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (c) ROC curves
    ax3 = fig.add_subplot(gs[0, 2])
    fpr_if, tpr_if = roc_curve(y, iforest_scores)
    auc_if = auc(fpr_if, tpr_if)
    ax3.plot(fpr_if, tpr_if, 'b-', linewidth=2,
            label=f'Isolation Forest (AUC={auc_if:.3f})')

    fpr_pca, tpr_pca = roc_curve(y, pca_errors)
    auc_pca = auc(fpr_pca, tpr_pca)
    ax3.plot(fpr_pca, tpr_pca, 'r--', linewidth=2,
            label=f'PCA Recon. Error (AUC={auc_pca:.3f})')

    ax3.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('(c) ROC Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (d) Anomaly heatmap (all parameters)
    ax4 = fig.add_subplot(gs[1, 0])
    # Normalize and show anomalies vs normal means
    normal_mean = X[normal_mask].mean(axis=0)
    normal_std = X[normal_mask].std(axis=0) + 1e-10
    anomaly_z = np.abs((X[anomaly_mask] - normal_mean) / normal_std)
    im = ax4.imshow(anomaly_z.T, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=5)
    ax4.set_yticks(range(8))
    ax4.set_yticklabels(PARAM_NAMES, fontsize=8)
    ax4.set_xlabel('Anomaly Sample Index')
    ax4.set_title('(d) Anomaly Z-Score Heatmap')
    plt.colorbar(im, ax=ax4, label='|Z-score|')

    # (e) Scatter colored by anomaly score
    ax5 = fig.add_subplot(gs[1, 1])
    sc = ax5.scatter(X[:, 4], X[:, 5], c=iforest_scores, cmap='RdYlGn_r',
                    s=20, alpha=0.6)
    ax5.scatter(X[anomaly_mask, 4], X[anomaly_mask, 5], s=60,
               facecolors='none', edgecolors='red', linewidths=1.5)
    plt.colorbar(sc, ax=ax5, label='Anomaly Score')
    ax5.set_xlabel(PARAM_NAMES[4])
    ax5.set_ylabel(PARAM_NAMES[5])
    ax5.set_title('(e) Ron vs Freq (colored by score)')
    ax5.grid(True, alpha=0.3)

    # (f) Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    # Compute metrics at threshold=0.5
    pred_if = (iforest_scores > 0.5).astype(int)
    tp = np.sum((pred_if == 1) & (y == 1))
    fp = np.sum((pred_if == 1) & (y == 0))
    fn = np.sum((pred_if == 0) & (y == 1))
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    summary = [
        ['Metric', 'Value'],
        ['Method', 'Isolation Forest'],
        ['Trees', '100'],
        ['Max Samples', '256'],
        ['Normal Samples', f'{int(normal_mask.sum())}'],
        ['Anomaly Samples', f'{int(anomaly_mask.sum())}'],
        ['AUC (IF)', f'{auc_if:.3f}'],
        ['AUC (PCA)', f'{auc_pca:.3f}'],
        ['Precision @0.5', f'{precision:.3f}'],
        ['Recall @0.5', f'{recall:.3f}'],
        ['F1 @0.5', f'{f1:.3f}'],
    ]
    table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax6.set_title('(f) Detection Performance', fontsize=12, pad=15)

    fig.text(0.5, 0.01,
             'Isolation Forest from scratch | PCA reconstruction error | '
             'ROC/AUC analysis | Multi-parameter ATE data',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Anomaly Detection for Semiconductor Test Data")
    print("Portfolio: Advantest — ML/Data Science")
    print("=" * 60)

    print("\nGenerating ATE test data...")
    X, y = generate_test_dataset(n_normal=800, n_anomaly=50)
    print(f"  Total: {len(y)} ({int(sum(y==0))} normal, {int(sum(y==1))} anomaly)")

    print("\nTraining Isolation Forest (100 trees)...")
    iforest = IsolationForest(n_estimators=100, max_samples=256)
    iforest.fit(X[y == 0])  # Train on normal data only
    scores = iforest.anomaly_score(X)
    pred = (scores > 0.5).astype(int)
    print(f"  Detected anomalies: {pred.sum()} (actual: {int(y.sum())})")

    print("\nPCA anomaly detection (4 components)...")
    pca = PCAAnomalyDetector(n_components=4)
    pca.fit(X[y == 0])
    pca_errors = pca.reconstruction_error(X)
    print(f"  Explained variance (top 4): {pca.explained_var[:4].sum():.3f}")

    # ROC/AUC
    fpr_if, tpr_if = roc_curve(y, scores)
    print(f"\n  IF AUC: {auc(fpr_if, tpr_if):.3f}")
    fpr_pca, tpr_pca = roc_curve(y, pca_errors)
    print(f"  PCA AUC: {auc(fpr_pca, tpr_pca):.3f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "anomaly_detection_results.png")
    print("\nPlotting results...")
    plot_results(X, y, scores, pca_errors, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
