#!/usr/bin/env python3
"""
GPU-Accelerated Neural Network from Scratch
=============================================
Portfolio: NVIDIA Japan — ML on GPU / CUDA

Implements a fully-connected neural network with GPU-accelerated
forward/backward pass using CuPy. CPU fallback with NumPy.

ML + GPU Concepts:
  - Matrix-multiply based forward/backward (GPU-friendly)
  - Mini-batch SGD with Adam optimizer
  - Xavier/He initialization
  - Dropout regularization
  - GPU vs CPU training speed comparison
  - Synthetic semiconductor yield classification

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# GPU/CPU backend abstraction
try:
    import cupy as cp
    HAS_GPU = True
    GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
except (ImportError, Exception):
    HAS_GPU = False
    GPU_NAME = "N/A"

def get_backend(use_gpu=False):
    """Return numpy-like backend (cupy or numpy)."""
    if use_gpu and HAS_GPU:
        return cp, 'GPU'
    return np, 'CPU'

# ============================================================
# Synthetic Dataset: Semiconductor Yield Classification
# ============================================================
def generate_dataset(n_samples=5000, n_features=20, n_classes=4):
    """Multi-class classification: yield grade from process features."""
    np.random.seed(42)
    # Cluster centers for each class
    centers = np.random.randn(n_classes, n_features) * 2
    X = []
    y = []
    for c in range(n_classes):
        n_c = n_samples // n_classes
        X_c = np.random.randn(n_c, n_features) * 0.8 + centers[c]
        X.append(X_c)
        y.append(np.full(n_c, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int32)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

# ============================================================
# Neural Network Layers
# ============================================================
class DenseLayer:
    def __init__(self, in_dim, out_dim, xp=np):
        # He initialization
        self.W = xp.array(np.random.randn(in_dim, out_dim).astype(np.float32)
                          * np.sqrt(2.0 / in_dim))
        self.b = xp.zeros(out_dim, dtype=xp.float32)
        # Adam state
        self.mW = xp.zeros_like(self.W)
        self.vW = xp.zeros_like(self.W)
        self.mb = xp.zeros_like(self.b)
        self.vb = xp.zeros_like(self.b)

class NeuralNetwork:
    """Multi-layer fully-connected network with ReLU + Softmax."""

    def __init__(self, layer_dims, dropout_rate=0.2, use_gpu=False):
        self.xp, self.device = get_backend(use_gpu)
        self.layers = []
        self.dropout_rate = dropout_rate
        for i in range(len(layer_dims) - 1):
            self.layers.append(DenseLayer(layer_dims[i], layer_dims[i+1],
                                          self.xp))
        self.training = True

    def relu(self, x):
        return self.xp.maximum(0, x)

    def softmax(self, x):
        shifted = x - self.xp.max(x, axis=1, keepdims=True)
        exp_x = self.xp.exp(shifted)
        return exp_x / self.xp.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass: store activations for backprop."""
        self.activations = [X]
        self.pre_activations = []
        out = X
        for i, layer in enumerate(self.layers):
            z = out @ layer.W + layer.b
            self.pre_activations.append(z)
            if i < len(self.layers) - 1:
                out = self.relu(z)
                # Dropout
                if self.training and self.dropout_rate > 0:
                    mask = (self.xp.random.rand(*out.shape) > self.dropout_rate).astype(
                        self.xp.float32)
                    out = out * mask / (1 - self.dropout_rate)
            else:
                out = self.softmax(z)
            self.activations.append(out)
        return out

    def compute_loss(self, probs, y, l2_lambda=1e-4):
        """Cross-entropy loss with L2 regularization."""
        N = len(y)
        # Use integer indexing for cross-entropy
        log_probs = self.xp.log(probs[self.xp.arange(N), y] + 1e-10)
        ce_loss = -self.xp.mean(log_probs)
        # L2
        l2 = sum(float(self.xp.sum(l.W**2)) for l in self.layers) * l2_lambda / 2
        return float(ce_loss) + l2

    def backward(self, probs, y):
        """Backpropagation."""
        N = len(y)
        xp = self.xp

        # Softmax gradient
        d = probs.copy()
        d[xp.arange(N), y] -= 1
        d /= N

        self.grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = self.activations[i]
            dW = a_prev.T @ d
            db = xp.sum(d, axis=0)
            self.grads.insert(0, (dW, db))

            if i > 0:
                d = d @ self.layers[i].W.T
                # ReLU backward
                d = d * (self.pre_activations[i-1] > 0).astype(xp.float32)

    def adam_update(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
        """Adam optimizer update."""
        xp = self.xp
        for layer, (dW, db) in zip(self.layers, self.grads):
            layer.mW = beta1 * layer.mW + (1 - beta1) * dW
            layer.vW = beta2 * layer.vW + (1 - beta2) * dW**2
            mW_hat = layer.mW / (1 - beta1**t)
            vW_hat = layer.vW / (1 - beta2**t)
            layer.W -= lr * mW_hat / (xp.sqrt(vW_hat) + eps)

            layer.mb = beta1 * layer.mb + (1 - beta1) * db
            layer.vb = beta2 * layer.vb + (1 - beta2) * db**2
            mb_hat = layer.mb / (1 - beta1**t)
            vb_hat = layer.vb / (1 - beta2**t)
            layer.b -= lr * mb_hat / (xp.sqrt(vb_hat) + eps)

    def predict(self, X):
        self.training = False
        probs = self.forward(X)
        self.training = True
        if HAS_GPU and self.device == 'GPU':
            return cp.asnumpy(self.xp.argmax(probs, axis=1))
        return self.xp.argmax(probs, axis=1)

# ============================================================
# Training Loop
# ============================================================
def train(model, X_train, y_train, X_val, y_val,
          epochs=30, batch_size=128, lr=0.001):
    """Train with mini-batch Adam."""
    xp = model.xp
    n = len(y_train)

    # Transfer data to device
    if model.device == 'GPU':
        X_tr = cp.array(X_train)
        y_tr = cp.array(y_train)
        X_v = cp.array(X_val)
        y_v = cp.array(y_val)
    else:
        X_tr, y_tr = X_train, y_train
        X_v, y_v = X_val, y_val

    history = {'train_loss': [], 'train_acc': [], 'val_acc': [],
               'epoch_time': []}

    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        idx = xp.random.permutation(n)
        epoch_loss = 0
        n_batches = 0
        model.training = True

        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            X_batch = X_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch)
            model.backward(probs, y_batch)
            model.adam_update(lr=lr, t=epoch*((n+batch_size-1)//batch_size) + n_batches + 1)

            epoch_loss += loss
            n_batches += 1

        if model.device == 'GPU':
            cp.cuda.Stream.null.synchronize()
        epoch_time = time.perf_counter() - t_epoch

        # Metrics
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_v)
        if model.device == 'GPU':
            train_acc = float(np.mean(train_pred == cp.asnumpy(y_tr)))
            val_acc = float(np.mean(val_pred == cp.asnumpy(y_v)))
        else:
            train_acc = float(np.mean(train_pred == y_tr))
            val_acc = float(np.mean(val_pred == y_v))

        history['train_loss'].append(epoch_loss / n_batches)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if (epoch + 1) % 5 == 0:
            print(f"  [{model.device}] Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f} "
                  f"train={train_acc:.3f} val={val_acc:.3f} time={epoch_time*1000:.0f}ms")

    return history

# ============================================================
# Visualization
# ============================================================
def plot_results(hist_cpu, hist_gpu, X_test, y_test, model_cpu, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('GPU-Accelerated Neural Network Training (NVIDIA)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # (a) Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist_cpu['train_loss'], 'b-', linewidth=2, label='CPU')
    if hist_gpu:
        ax1.plot(hist_gpu['train_loss'], 'r--', linewidth=2, label='GPU')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Validation accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(hist_cpu['val_acc'], 'b-', linewidth=2, label='CPU Val Acc')
    if hist_gpu:
        ax2.plot(hist_gpu['val_acc'], 'r--', linewidth=2, label='GPU Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) Training speed comparison
    ax3 = fig.add_subplot(gs[0, 2])
    cpu_time = np.mean(hist_cpu['epoch_time']) * 1000
    labels = ['CPU']
    times = [cpu_time]
    colors = ['steelblue']
    if hist_gpu:
        gpu_time = np.mean(hist_gpu['epoch_time']) * 1000
        labels.append('GPU')
        times.append(gpu_time)
        colors.append('limegreen')
    bars = ax3.bar(labels, times, color=colors, alpha=0.8)
    for bar, t in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', fontsize=12, fontweight='bold')
    if hist_gpu:
        speedup = cpu_time / gpu_time
        ax3.set_title(f'(c) Epoch Time ({speedup:.1f}x speedup)')
    else:
        ax3.set_title('(c) Epoch Time')
    ax3.set_ylabel('Time per Epoch [ms]')
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) Confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    y_pred = model_cpu.predict(X_test)
    n_classes = 4
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1
    im = ax4.imshow(cm, cmap='Blues')
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax4.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=12, color=color)
    grade_names = ['Grade A', 'Grade B', 'Grade C', 'Grade D']
    ax4.set_xticks(range(n_classes))
    ax4.set_xticklabels(grade_names, fontsize=9)
    ax4.set_yticks(range(n_classes))
    ax4.set_yticklabels(grade_names, fontsize=9)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('(d) Confusion Matrix')

    # (e) Per-epoch timing breakdown
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot([t*1000 for t in hist_cpu['epoch_time']], 'b-o', markersize=4,
            label='CPU', linewidth=1.5)
    if hist_gpu:
        ax5.plot([t*1000 for t in hist_gpu['epoch_time']], 'r-^', markersize=4,
                label='GPU', linewidth=1.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Time [ms]')
    ax5.set_title('(e) Per-Epoch Training Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # (f) Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    final_cpu_acc = hist_cpu['val_acc'][-1]
    final_gpu_acc = hist_gpu['val_acc'][-1] if hist_gpu else 'N/A'
    summary = [
        ['', 'CPU', 'GPU' if hist_gpu else 'N/A'],
        ['Backend', 'NumPy', 'CuPy/CUDA' if hist_gpu else '-'],
        ['Final Val Acc', f'{final_cpu_acc:.3f}',
         f'{final_gpu_acc:.3f}' if hist_gpu else '-'],
        ['Avg Epoch Time', f'{np.mean(hist_cpu["epoch_time"])*1000:.1f}ms',
         f'{np.mean(hist_gpu["epoch_time"])*1000:.1f}ms' if hist_gpu else '-'],
        ['Total Time', f'{sum(hist_cpu["epoch_time"]):.1f}s',
         f'{sum(hist_gpu["epoch_time"]):.1f}s' if hist_gpu else '-'],
        ['', '', ''],
        ['Architecture', '20-128-64-32-4', ''],
        ['Optimizer', 'Adam (lr=0.001)', ''],
        ['Dropout', '0.2', ''],
        ['Batch Size', '128', ''],
        ['Samples', '5000', ''],
        ['GPU', GPU_NAME if HAS_GPU else 'N/A', ''],
    ]
    table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax6.set_title('(f) Training Summary', fontsize=12, pad=15)

    fig.text(0.5, 0.01,
             'Neural Network from scratch | Adam optimizer | '
             'Dropout regularization | GPU/CPU comparison via CuPy',
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
    print("GPU-Accelerated Neural Network from Scratch")
    print("Portfolio: NVIDIA Japan — ML on GPU")
    print("=" * 60)

    print(f"\nGPU: {GPU_NAME if HAS_GPU else 'Not available'}")

    # Dataset
    print("\nGenerating dataset (5000 samples, 20 features, 4 classes)...")
    X, y = generate_dataset(5000, 20, 4)
    n_train = int(0.8 * len(y))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # CPU Training
    print("\n--- CPU Training (NumPy) ---")
    model_cpu = NeuralNetwork([20, 128, 64, 32, 4], dropout_rate=0.2, use_gpu=False)
    hist_cpu = train(model_cpu, X_train, y_train, X_test, y_test,
                     epochs=30, batch_size=128, lr=0.001)

    # GPU Training
    hist_gpu = None
    if HAS_GPU:
        print("\n--- GPU Training (CuPy/CUDA) ---")
        model_gpu = NeuralNetwork([20, 128, 64, 32, 4], dropout_rate=0.2, use_gpu=True)
        hist_gpu = train(model_gpu, X_train, y_train, X_test, y_test,
                         epochs=30, batch_size=128, lr=0.001)
    else:
        print("\n[Skip GPU training — CuPy not available]")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "cuda_neural_net_results.png")
    print("\nPlotting results...")
    plot_results(hist_cpu, hist_gpu, X_test, y_test, model_cpu, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
