#!/usr/bin/env python3
"""
Gradient Boosting for Semiconductor Yield Prediction
======================================================
Portfolio: Renesas — ML/Data Science

Predicts wafer yield from fab process parameters using
Gradient Boosting Decision Trees (GBDT) implemented from scratch.

ML Concepts:
  - Gradient Boosting from scratch (sequential ensemble)
  - Regression trees with MSE split criterion
  - Learning rate, early stopping
  - Partial dependence plots (PDP)
  - Wafer map yield pattern analysis

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
# Synthetic Fab Process Dataset
# ============================================================
FEATURE_NAMES = [
    'Implant Dose [cm^-2]', 'Anneal Temp [C]', 'Oxide Thickness [nm]',
    'Etch Time [s]', 'CMP Pressure [psi]', 'Litho Focus [um]',
    'Metal Dep Rate [nm/s]', 'Clean Duration [s]'
]

def generate_yield_dataset(n_wafers=600):
    """Generate synthetic fab yield data."""
    n = n_wafers
    X = np.zeros((n, 8))
    X[:, 0] = np.random.uniform(1e12, 1e14, n)     # Implant dose
    X[:, 1] = np.random.uniform(800, 1100, n)       # Anneal temp
    X[:, 2] = np.random.uniform(2, 20, n)           # Oxide thickness
    X[:, 3] = np.random.uniform(30, 120, n)         # Etch time
    X[:, 4] = np.random.uniform(1, 8, n)            # CMP pressure
    X[:, 5] = np.random.uniform(-0.5, 0.5, n)       # Litho focus offset
    X[:, 6] = np.random.uniform(0.5, 5, n)          # Metal dep rate
    X[:, 7] = np.random.uniform(30, 180, n)         # Clean duration

    # Yield model: nonlinear with interactions
    x_norm = np.zeros_like(X)
    for i in range(8):
        x_norm[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std() + 1e-10)

    # Base yield
    y = 95.0 * np.ones(n)
    # Implant dose: sweet spot
    y -= 8 * (x_norm[:, 0] - 0.2)**2
    # Anneal temp: higher is generally better but saturates
    y += 3 * np.tanh(x_norm[:, 1])
    # Oxide thickness: thin = better for advanced nodes
    y -= 2 * x_norm[:, 2]
    # Litho focus: must be near zero
    y -= 15 * x_norm[:, 5]**2
    # CMP pressure x Etch time interaction
    y -= 3 * x_norm[:, 4] * x_norm[:, 3]
    # Metal dep rate: moderate is best
    y -= 4 * (x_norm[:, 6] - 0.1)**2
    # Noise
    y += np.random.randn(n) * 2
    y = np.clip(y, 0, 100)

    return X, y

# ============================================================
# Regression Tree (for GBDT base learner)
# ============================================================
class RegressionTree:
    def __init__(self, max_depth=4, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        n = len(y)
        parent_mse = self._mse(y)

        n_features = X.shape[1]
        feat_indices = np.random.choice(n_features, max(1, n_features // 2),
                                        replace=False)

        for feat in feat_indices:
            thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 15))
            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) < self.min_samples_leaf or \
                   len(right) < self.min_samples_leaf:
                    continue
                gain = parent_mse - self._mse(left) - self._mse(right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or np.std(y) < 1e-6):
            return {'leaf': True, 'value': np.mean(y)}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'value': np.mean(y)}

        mask = X[:, feat] <= thresh
        return {
            'leaf': False, 'feature': feat, 'threshold': thresh,
            'left': self._build(X[mask], y[mask], depth + 1),
            'right': self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ============================================================
# Gradient Boosting (from scratch)
# ============================================================
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=4,
                 subsample=0.8):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.trees = []
        self.train_scores = []
        self.val_scores = []
        self.initial_pred = None

    def fit(self, X, y, X_val=None, y_val=None):
        n = len(y)
        self.initial_pred = np.mean(y)
        F = np.full(n, self.initial_pred)
        if X_val is not None:
            F_val = np.full(len(y_val), self.initial_pred)

        for i in range(self.n_estimators):
            # Negative gradient (residuals for MSE loss)
            residuals = y - F

            # Subsample
            idx = np.random.choice(n, int(n * self.subsample), replace=False)

            # Fit tree to residuals
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X[idx], residuals[idx])
            self.trees.append(tree)

            # Update predictions
            F += self.lr * tree.predict(X)

            # Track scores
            train_rmse = np.sqrt(np.mean((y - F)**2))
            self.train_scores.append(train_rmse)

            if X_val is not None:
                F_val += self.lr * tree.predict(X_val)
                val_rmse = np.sqrt(np.mean((y_val - F_val)**2))
                self.val_scores.append(val_rmse)

    def predict(self, X):
        F = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F

    def feature_importance(self, X, y):
        """Permutation-based feature importance."""
        base_mse = np.mean((y - self.predict(X))**2)
        importances = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_mse = np.mean((y - self.predict(X_perm))**2)
            importances[j] = perm_mse - base_mse
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

# ============================================================
# Partial Dependence Plot
# ============================================================
def partial_dependence(model, X, feature_idx, n_grid=50):
    """Compute partial dependence for one feature."""
    x_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    pd_values = np.zeros(n_grid)
    for i, val in enumerate(x_range):
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        pd_values[i] = np.mean(model.predict(X_mod))
    return x_range, pd_values

# ============================================================
# Visualization
# ============================================================
def plot_results(model, X_test, y_test, importances, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Gradient Boosting for Semiconductor Yield Prediction (Renesas)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    y_pred = model.predict(X_test)

    # (a) Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, s=20, alpha=0.5, c='steelblue')
    lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
    ax1.plot(lims, lims, 'r--', linewidth=2)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
    ax1.set_xlabel('Actual Yield [%]')
    ax1.set_ylabel('Predicted Yield [%]')
    ax1.set_title(f'(a) Prediction (RMSE={rmse:.2f}, R2={r2:.3f})')
    ax1.grid(True, alpha=0.3)

    # (b) Training curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(model.train_scores, 'b-', label='Train RMSE')
    if model.val_scores:
        ax2.plot(model.val_scores, 'r--', label='Val RMSE')
    ax2.set_xlabel('Boosting Iteration')
    ax2.set_ylabel('RMSE')
    ax2.set_title('(b) Boosting Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) Feature Importance
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_idx = np.argsort(importances)
    ax3.barh(range(len(importances)), importances[sorted_idx], color='steelblue')
    ax3.set_yticks(range(len(importances)))
    ax3.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=8)
    ax3.set_xlabel('Permutation Importance')
    ax3.set_title('(c) Feature Importance')
    ax3.grid(True, alpha=0.3, axis='x')

    # (d-f) Partial Dependence Plots (top 3 features)
    top3 = np.argsort(importances)[-3:][::-1]
    for panel, feat_idx in enumerate(top3):
        ax = fig.add_subplot(gs[1, panel])
        x_range, pd_vals = partial_dependence(model, X_test, feat_idx)
        ax.plot(x_range, pd_vals, 'b-', linewidth=2)
        ax.fill_between(x_range, pd_vals - 1, pd_vals + 1, alpha=0.2)
        # Rug plot
        ax.scatter(X_test[:, feat_idx], np.full(len(X_test), pd_vals.min()-0.5),
                  s=3, alpha=0.3, color='gray')
        ax.set_xlabel(FEATURE_NAMES[feat_idx])
        ax.set_ylabel('Predicted Yield [%]')
        ax.set_title(f'({chr(100+panel)}) PDP: {FEATURE_NAMES[feat_idx][:20]}')
        ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             'GBDT from scratch | Regression trees (MSE) | '
             'Permutation importance | Partial dependence plots',
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
    print("Gradient Boosting for Semiconductor Yield Prediction")
    print("Portfolio: Renesas — ML/Data Science")
    print("=" * 60)

    print("\nGenerating fab dataset...")
    X, y = generate_yield_dataset(600)
    print(f"  Wafers: {len(y)}, Yield range: {y.min():.1f}-{y.max():.1f}%")

    n_train = int(0.8 * len(y))
    idx = np.random.permutation(len(y))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    print("\nTraining Gradient Boosting (150 trees)...")
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                       max_depth=4, subsample=0.8)
    model.fit(X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
    print(f"  Test RMSE: {rmse:.2f}%")
    print(f"  Test R2: {r2:.3f}")

    print("\nComputing feature importance...")
    importances = model.feature_importance(X_test, y_test)
    top3 = np.argsort(importances)[-3:][::-1]
    for i in top3:
        print(f"  {FEATURE_NAMES[i]}: {importances[i]:.4f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "yield_prediction_results.png")
    print("\nPlotting results...")
    plot_results(model, X_test, y_test, importances, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
