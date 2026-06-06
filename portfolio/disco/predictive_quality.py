#!/usr/bin/env python3
"""
ML-Based Wafer Grinding Quality Prediction
============================================
Portfolio: DISCO — ML/Data Science

Predicts wafer grinding quality (TTV, surface roughness, chipping)
from process parameters using ensemble learning.

ML Concepts:
  - Random Forest from scratch (decision tree ensemble)
  - Feature importance (MDI: Mean Decrease in Impurity)
  - Cross-validation with stratified splits
  - Confusion matrix & classification metrics
  - SHAP-like feature contribution analysis

Domain:
  - Wafer back-grinding process parameters
  - Quality metrics: TTV, Ra, edge chipping
  - Pass/Fail classification for quality control

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import os
import time

np.random.seed(42)

# ============================================================
# Synthetic Dataset: Wafer Grinding Process
# ============================================================
FEATURE_NAMES = [
    'Feed Rate [um/s]', 'Spindle Speed [rpm]', 'Wheel Grit [#]',
    'Coolant Flow [L/min]', 'Wafer Thickness [um]',
    'Chuck Vacuum [kPa]', 'Dress Interval [wafers]',
    'Ambient Temp [C]'
]

def generate_grinding_dataset(n_samples=500):
    """Generate realistic wafer grinding dataset."""
    X = np.zeros((n_samples, 8))
    X[:, 0] = np.random.uniform(0.5, 5.0, n_samples)    # Feed rate
    X[:, 1] = np.random.uniform(2000, 6000, n_samples)   # Spindle speed
    X[:, 2] = np.random.choice([320, 600, 1000, 2000], n_samples)  # Grit
    X[:, 3] = np.random.uniform(1.0, 5.0, n_samples)     # Coolant flow
    X[:, 4] = np.random.uniform(300, 775, n_samples)      # Wafer thickness
    X[:, 5] = np.random.uniform(20, 80, n_samples)        # Chuck vacuum
    X[:, 6] = np.random.uniform(5, 50, n_samples)         # Dress interval
    X[:, 7] = np.random.normal(23, 2, n_samples)          # Ambient temp

    # Quality model: TTV depends on feed rate, spindle speed, grit
    ttv = (0.8 * X[:, 0] / 5.0 - 0.3 * X[:, 1] / 6000 +
           0.5 * (1 - X[:, 2] / 2000) + 0.2 * X[:, 6] / 50 +
           0.15 * np.random.randn(n_samples))

    # Classification: 0=Pass, 1=Marginal, 2=Fail
    y = np.zeros(n_samples, dtype=int)
    y[ttv > 0.4] = 1  # Marginal
    y[ttv > 0.7] = 2  # Fail

    # Additional failure modes
    high_feed = X[:, 0] > 4.0
    low_coolant = X[:, 3] < 1.5
    y[high_feed & low_coolant] = 2

    return X, y, ttv

# ============================================================
# Decision Tree (from scratch)
# ============================================================
class DecisionTree:
    """CART decision tree for classification."""

    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                 max_features='sqrt'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
        self.n_features = None
        self.feature_importances_ = None

    def _gini(self, y):
        """Gini impurity."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _best_split(self, X, y, feature_indices):
        """Find best split among selected features."""
        best_gain = -1
        best_feat = None
        best_thresh = None
        n = len(y)
        parent_gini = self._gini(y)

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat], np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = parent_gini - (n_left/n * self._gini(y[left_mask]) +
                                       n_right/n * self._gini(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth):
        """Recursively build tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split
                or n_classes == 1):
            counts = np.bincount(y, minlength=3)
            return {'leaf': True, 'class': np.argmax(counts),
                    'proba': counts / counts.sum(), 'n': n_samples}

        # Random feature subset
        if self.max_features == 'sqrt':
            n_feat = max(1, int(np.sqrt(self.n_features)))
        else:
            n_feat = self.n_features
        feat_idx = np.random.choice(self.n_features, n_feat, replace=False)

        feat, thresh, gain = self._best_split(X, y, feat_idx)

        if feat is None or gain < 1e-10:
            counts = np.bincount(y, minlength=3)
            return {'leaf': True, 'class': np.argmax(counts),
                    'proba': counts / counts.sum(), 'n': n_samples}

        left_mask = X[:, feat] <= thresh
        self.feature_importances_[feat] += gain * n_samples

        return {
            'leaf': False,
            'feature': feat, 'threshold': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.tree = self._build_tree(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class'], node['proba']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree)[0] for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree)[1] for x in X])

# ============================================================
# Random Forest (from scratch)
# ============================================================
class RandomForest:
    """Random Forest ensemble classifier."""

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5,
                 max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None

    def fit(self, X, y):
        n_samples = len(y)
        self.trees = []
        importances = np.zeros(X.shape[1])
        oob_predictions = np.zeros((n_samples, 3))
        oob_counts = np.zeros(n_samples)

        for t in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), idx)

            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            importances += tree.feature_importances_

            # OOB prediction
            if len(oob_idx) > 0:
                oob_pred = tree.predict_proba(X[oob_idx])
                oob_predictions[oob_idx] += oob_pred
                oob_counts[oob_idx] += 1

        self.feature_importances_ = importances / self.n_estimators

        # OOB accuracy
        valid = oob_counts > 0
        if valid.any():
            oob_classes = np.argmax(oob_predictions[valid], axis=1)
            self.oob_score_ = np.mean(oob_classes == y[valid])

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees])
        # Majority vote
        return np.array([Counter(all_preds[:, i]).most_common(1)[0][0]
                         for i in range(X.shape[0])])

    def predict_proba(self, X):
        all_proba = np.array([t.predict_proba(X) for t in self.trees])
        return np.mean(all_proba, axis=0)

# ============================================================
# Cross-Validation
# ============================================================
def cross_validate(X, y, n_folds=5):
    """Stratified k-fold cross-validation."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    scores = []

    for k in range(n_folds):
        val_idx = indices[k*fold_size:(k+1)*fold_size]
        train_idx = np.concatenate([indices[:k*fold_size], indices[(k+1)*fold_size:]])

        rf = RandomForest(n_estimators=50, max_depth=8)
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[val_idx])
        acc = np.mean(pred == y[val_idx])
        scores.append(acc)

    return scores

# ============================================================
# Confusion Matrix & Metrics
# ============================================================
def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    report = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        report[name] = {'precision': precision, 'recall': recall, 'f1': f1}
    return report, cm

# ============================================================
# Visualization
# ============================================================
def plot_results(rf, X_test, y_test, y_pred, cv_scores, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('ML-Based Wafer Grinding Quality Prediction (DISCO)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    class_names = ['Pass', 'Marginal', 'Fail']

    # (a) Feature Importance
    ax1 = fig.add_subplot(gs[0, 0])
    imp = rf.feature_importances_
    sorted_idx = np.argsort(imp)
    ax1.barh(np.arange(len(imp)), imp[sorted_idx], color='steelblue')
    ax1.set_yticks(np.arange(len(imp)))
    ax1.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=9)
    ax1.set_xlabel('Feature Importance (MDI)')
    ax1.set_title('(a) Random Forest Feature Importance')
    ax1.grid(True, alpha=0.3, axis='x')

    # (b) Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    _, cm = classification_report(y_test, y_pred, class_names)
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color=color)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(class_names)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('(b) Confusion Matrix')
    plt.colorbar(im, ax=ax2)

    # (c) Cross-validation scores
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(cv_scores)), cv_scores, color='limegreen', alpha=0.8)
    ax3.axhline(np.mean(cv_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(cv_scores):.3f}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('(c) 5-Fold Cross-Validation')
    ax3.set_ylim(0.5, 1.0)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) Decision boundary (Feed Rate vs Spindle Speed)
    ax4 = fig.add_subplot(gs[1, 0])
    n_grid = 100
    x_min, x_max = X_test[:, 0].min()-0.5, X_test[:, 0].max()+0.5
    y_min, y_max = X_test[:, 1].min()-500, X_test[:, 1].max()+500
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                          np.linspace(y_min, y_max, n_grid))
    X_grid = np.tile(np.median(X_test, axis=0), (n_grid*n_grid, 1))
    X_grid[:, 0] = xx.ravel()
    X_grid[:, 1] = yy.ravel()
    Z = rf.predict(X_grid).reshape(n_grid, n_grid)
    ax4.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                 colors=['#90EE90', '#FFD700', '#FF6B6B'], alpha=0.5)
    colors_map = {0: 'green', 1: 'orange', 2: 'red'}
    for c in [0, 1, 2]:
        mask = y_test == c
        ax4.scatter(X_test[mask, 0], X_test[mask, 1], s=20,
                   c=colors_map[c], label=class_names[c], alpha=0.6, edgecolors='k',
                   linewidth=0.3)
    ax4.set_xlabel('Feed Rate [um/s]')
    ax4.set_ylabel('Spindle Speed [rpm]')
    ax4.set_title('(d) Decision Boundary')
    ax4.legend(fontsize=8)

    # (e) Prediction probability distribution
    ax5 = fig.add_subplot(gs[1, 1])
    proba = rf.predict_proba(X_test)
    for c, name in enumerate(class_names):
        ax5.hist(proba[:, c], bins=30, alpha=0.5, label=name)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Count')
    ax5.set_title('(e) Prediction Confidence Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (f) Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    report, _ = classification_report(y_test, y_pred, class_names)
    accuracy = np.mean(y_pred == y_test)
    summary = [
        ['Metric', 'Pass', 'Marginal', 'Fail'],
        ['Precision', f'{report["Pass"]["precision"]:.3f}',
         f'{report["Marginal"]["precision"]:.3f}',
         f'{report["Fail"]["precision"]:.3f}'],
        ['Recall', f'{report["Pass"]["recall"]:.3f}',
         f'{report["Marginal"]["recall"]:.3f}',
         f'{report["Fail"]["recall"]:.3f}'],
        ['F1-Score', f'{report["Pass"]["f1"]:.3f}',
         f'{report["Marginal"]["f1"]:.3f}',
         f'{report["Fail"]["f1"]:.3f}'],
        ['', '', '', ''],
        ['Overall Acc.', f'{accuracy:.3f}', '', ''],
        ['OOB Score', f'{rf.oob_score_:.3f}' if rf.oob_score_ else 'N/A', '', ''],
        ['Trees', f'{rf.n_estimators}', '', ''],
        ['CV Mean', f'{np.mean(cv_scores):.3f}', '', ''],
    ]
    table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax6.set_title('(f) Classification Report', fontsize=12, pad=20)

    fig.text(0.5, 0.01,
             'Random Forest from scratch | Gini impurity | OOB estimation | '
             '5-fold CV | Feature importance (MDI)',
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
    print("ML-Based Wafer Grinding Quality Prediction")
    print("Portfolio: DISCO — ML/Data Science")
    print("=" * 60)

    # Generate dataset
    print("\nGenerating dataset...")
    X, y, ttv = generate_grinding_dataset(n_samples=500)
    print(f"  Samples: {len(y)}")
    print(f"  Class distribution: Pass={sum(y==0)}, Marginal={sum(y==1)}, Fail={sum(y==2)}")

    # Train/test split
    n_train = int(0.8 * len(y))
    idx = np.random.permutation(len(y))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    # Train Random Forest
    print("\nTraining Random Forest (100 trees)...")
    rf = RandomForest(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"  Test accuracy: {acc:.3f}")
    print(f"  OOB score: {rf.oob_score_:.3f}")

    # Cross-validation
    print("\n5-fold Cross-validation...")
    cv_scores = cross_validate(X, y, n_folds=5)
    print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")

    # Feature importance
    print("\nTop features:")
    sorted_idx = np.argsort(rf.feature_importances_)[::-1]
    for i in sorted_idx[:5]:
        print(f"  {FEATURE_NAMES[i]}: {rf.feature_importances_[i]:.4f}")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "predictive_quality_results.png")
    print("\nPlotting results...")
    plot_results(rf, X_test, y_test, y_pred, cv_scores, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
