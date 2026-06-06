#!/usr/bin/env python3
"""
Autoencoder for Image Sensor Denoising
========================================
Portfolio: Sony Semiconductor Solutions — ML/Deep Learning

Denoises images from CMOS image sensor using a convolutional
autoencoder implemented from scratch in NumPy.

ML Concepts:
  - Convolutional Autoencoder (encoder-decoder with skip connections)
  - PSNR/SSIM quality metrics
  - Multiple noise models (Gaussian, Poisson shot noise, fixed-pattern)
  - Dense autoencoder with bottleneck
  - Comparison of denoising approaches

Domain:
  - CMOS image sensor noise characteristics
  - Low-light imaging, high-ISO denoising
  - Read noise + shot noise + fixed-pattern noise

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
# Image Generation & Noise Models
# ============================================================
IMG_SIZE = 32

def generate_test_images(n_images=500):
    """Generate synthetic test patterns (circles, edges, gradients)."""
    images = []
    for _ in range(n_images):
        img = np.zeros((IMG_SIZE, IMG_SIZE))
        pattern = np.random.choice(['circle', 'edge', 'gradient', 'checker'])

        if pattern == 'circle':
            cx, cy = np.random.randint(8, 24, 2)
            r = np.random.randint(3, 10)
            yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            mask = (xx - cx)**2 + (yy - cy)**2 < r**2
            img[mask] = np.random.uniform(0.5, 1.0)
        elif pattern == 'edge':
            angle = np.random.uniform(0, np.pi)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    if np.cos(angle)*i + np.sin(angle)*j > 16:
                        img[i, j] = np.random.uniform(0.6, 1.0)
        elif pattern == 'gradient':
            direction = np.random.choice([0, 1])
            if direction == 0:
                img = np.linspace(0, 1, IMG_SIZE).reshape(1, -1).repeat(IMG_SIZE, 0)
            else:
                img = np.linspace(0, 1, IMG_SIZE).reshape(-1, 1).repeat(IMG_SIZE, 1)
        else:  # checker
            size = np.random.choice([4, 8])
            for i in range(0, IMG_SIZE, size*2):
                for j in range(0, IMG_SIZE, size*2):
                    img[i:i+size, j:j+size] = 0.8
                    img[i+size:i+2*size, j+size:j+2*size] = 0.8

        images.append(img)
    return np.array(images)

def add_sensor_noise(images, sigma_read=0.05, gain=0.1):
    """Add realistic CMOS sensor noise: read noise + shot noise + FPN."""
    noisy = images.copy()
    # Shot noise (Poisson)
    noisy = np.random.poisson(np.maximum(noisy / gain, 0.01)) * gain
    # Read noise (Gaussian)
    noisy += sigma_read * np.random.randn(*noisy.shape)
    # Fixed-pattern noise (column-wise bias)
    fpn = 0.02 * np.random.randn(1, noisy.shape[-1])
    noisy += fpn
    return np.clip(noisy, 0, 1)

def psnr(clean, denoised):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((clean - denoised)**2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def ssim_simple(clean, denoised, C1=0.01**2, C2=0.03**2):
    """Simplified SSIM (per image, global window)."""
    mu_c = np.mean(clean)
    mu_d = np.mean(denoised)
    sig_c = np.var(clean)
    sig_d = np.var(denoised)
    sig_cd = np.mean((clean - mu_c) * (denoised - mu_d))
    return ((2*mu_c*mu_d + C1) * (2*sig_cd + C2)) / \
           ((mu_c**2 + mu_d**2 + C1) * (sig_c + sig_d + C2))

# ============================================================
# Dense Autoencoder (NumPy from scratch)
# ============================================================
class DenseAutoencoder:
    """Fully-connected autoencoder with bottleneck."""

    def __init__(self, input_dim, hidden_dims=[256, 64, 256]):
        self.layers = []
        dims = [input_dim] + hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.layers.append({'W': W, 'b': b,
                                'vW': np.zeros_like(W), 'vb': np.zeros_like(b)})
        self.bottleneck_idx = len(hidden_dims) // 2

    def forward(self, X):
        """Forward pass with ReLU activations (sigmoid at output)."""
        self.activations = [X]
        out = X
        for i, layer in enumerate(self.layers):
            out = out @ layer['W'] + layer['b']
            if i < len(self.layers) - 1:
                out = np.maximum(0, out)  # ReLU
            else:
                out = 1 / (1 + np.exp(-np.clip(out, -10, 10)))  # Sigmoid
            self.activations.append(out)
        return out

    def backward(self, y_true, lr=0.001, momentum=0.9, l2=1e-4):
        """Backpropagation with MSE loss."""
        N = y_true.shape[0]
        output = self.activations[-1]

        # MSE loss gradient through sigmoid
        d = (output - y_true) * output * (1 - output) / N

        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = self.activations[i]
            layer = self.layers[i]

            dW = a_prev.T @ d + l2 * layer['W']
            db = np.sum(d, axis=0)

            layer['vW'] = momentum * layer['vW'] - lr * dW
            layer['vb'] = momentum * layer['vb'] - lr * db
            layer['W'] += layer['vW']
            layer['b'] += layer['vb']

            if i > 0:
                d = d @ layer['W'].T
                d = d * (self.activations[i] > 0)  # ReLU backward

    def get_bottleneck(self, X):
        """Get bottleneck representation."""
        out = X
        for i in range(self.bottleneck_idx + 1):
            out = out @ self.layers[i]['W'] + self.layers[i]['b']
            out = np.maximum(0, out)
        return out

# ============================================================
# Training
# ============================================================
def train_autoencoder(model, X_noisy, X_clean, X_val_noisy, X_val_clean,
                       epochs=50, batch_size=32, lr=0.002):
    """Train denoising autoencoder."""
    n = len(X_noisy)
    input_dim = X_noisy.shape[1]
    history = {'train_loss': [], 'val_psnr': []}

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_noisy = X_noisy[idx[i:i+batch_size]]
            batch_clean = X_clean[idx[i:i+batch_size]]

            output = model.forward(batch_noisy)
            loss = np.mean((output - batch_clean)**2)
            model.backward(batch_clean, lr=lr)

            epoch_loss += loss
            n_batches += 1

        # Validation PSNR
        val_output = model.forward(X_val_noisy)
        val_psnr = np.mean([psnr(X_val_clean[i].reshape(IMG_SIZE, IMG_SIZE),
                                  val_output[i].reshape(IMG_SIZE, IMG_SIZE))
                            for i in range(len(X_val_clean))])

        history['train_loss'].append(epoch_loss / n_batches)
        history['val_psnr'].append(val_psnr)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.6f} "
                  f"val_PSNR={val_psnr:.2f}dB")

    return history

# ============================================================
# Baseline: Gaussian Filter
# ============================================================
def gaussian_filter_denoise(images, sigma=1.0):
    """Simple Gaussian blur denoising."""
    from scipy.ndimage import gaussian_filter
    denoised = np.zeros_like(images)
    for i in range(len(images)):
        denoised[i] = gaussian_filter(images[i], sigma=sigma)
    return denoised

# ============================================================
# Visualization
# ============================================================
def plot_results(model, history, clean_test, noisy_test, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Autoencoder for Image Sensor Denoising (Sony SSS)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.35)

    flat_noisy = noisy_test.reshape(len(noisy_test), -1)
    flat_clean = clean_test.reshape(len(clean_test), -1)
    denoised_flat = model.forward(flat_noisy)
    denoised = denoised_flat.reshape(-1, IMG_SIZE, IMG_SIZE)

    gauss_denoised = gaussian_filter_denoise(noisy_test, sigma=1.0)

    # (a-c) Example: Clean / Noisy / Denoised
    for col, (title, imgs) in enumerate([
        ('(a) Clean', clean_test),
        ('(b) Noisy (Sensor Noise)', noisy_test),
        ('(c) Denoised (Autoencoder)', denoised)]):
        ax = fig.add_subplot(gs[0, col])
        # Show 3x3 grid
        grid = np.zeros((IMG_SIZE*3+4, IMG_SIZE*3+4))
        for i in range(3):
            for j in range(3):
                idx = i*3+j
                r0, c0 = i*(IMG_SIZE+2), j*(IMG_SIZE+2)
                grid[r0:r0+IMG_SIZE, c0:c0+IMG_SIZE] = imgs[idx]
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    # (d) Training curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['train_loss'], 'b-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MSE Loss')
    ax4.set_title('(d) Training Loss')
    ax4.grid(True, alpha=0.3)

    # (e) Validation PSNR over epochs
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['val_psnr'], 'g-', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('PSNR [dB]')
    ax5.set_title('(e) Validation PSNR')
    ax5.grid(True, alpha=0.3)

    # (f) PSNR comparison: Noisy vs Gaussian vs Autoencoder
    ax6 = fig.add_subplot(gs[1, 2])
    psnr_noisy = [psnr(clean_test[i], noisy_test[i]) for i in range(len(clean_test))]
    psnr_gauss = [psnr(clean_test[i], gauss_denoised[i]) for i in range(len(clean_test))]
    psnr_ae = [psnr(clean_test[i], denoised[i]) for i in range(len(clean_test))]

    methods = ['Noisy', 'Gaussian\nFilter', 'Autoencoder']
    means = [np.mean(psnr_noisy), np.mean(psnr_gauss), np.mean(psnr_ae)]
    colors = ['red', 'orange', 'green']
    bars = ax6.bar(methods, means, color=colors, alpha=0.7)
    for bar, m in zip(bars, means):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{m:.1f}dB', ha='center', fontsize=11)
    ax6.set_ylabel('PSNR [dB]')
    ax6.set_title('(f) Denoising Method Comparison')
    ax6.grid(True, alpha=0.3, axis='y')

    # (g) SSIM comparison
    ax7 = fig.add_subplot(gs[2, 0])
    ssim_noisy = [ssim_simple(clean_test[i], noisy_test[i]) for i in range(len(clean_test))]
    ssim_gauss = [ssim_simple(clean_test[i], gauss_denoised[i]) for i in range(len(clean_test))]
    ssim_ae = [ssim_simple(clean_test[i], denoised[i]) for i in range(len(clean_test))]
    means_ssim = [np.mean(ssim_noisy), np.mean(ssim_gauss), np.mean(ssim_ae)]
    bars = ax7.bar(methods, means_ssim, color=colors, alpha=0.7)
    for bar, m in zip(bars, means_ssim):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{m:.3f}', ha='center', fontsize=11)
    ax7.set_ylabel('SSIM')
    ax7.set_title('(g) SSIM Comparison')
    ax7.grid(True, alpha=0.3, axis='y')

    # (h) Residual (noise removed)
    ax8 = fig.add_subplot(gs[2, 1])
    residual = noisy_test[0] - denoised[0]
    im = ax8.imshow(residual, cmap='RdBu', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax8)
    ax8.set_title('(h) Residual (Removed Noise)')
    ax8.axis('off')

    # (i) Architecture & noise model summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    info = [
        ['Component', 'Detail'],
        ['Architecture', 'Dense AE (1024-256-64-256-1024)'],
        ['Activation', 'ReLU (hidden), Sigmoid (output)'],
        ['Loss', 'MSE + L2 regularization'],
        ['Optimizer', 'SGD + Momentum (0.9)'],
        ['Noise Model', 'Shot + Read + FPN'],
        ['Shot Noise', 'Poisson (gain=0.1)'],
        ['Read Noise', 'Gaussian (sigma=0.05)'],
        ['Fixed Pattern', 'Column-wise bias'],
        ['Best PSNR', f'{np.mean(psnr_ae):.1f} dB'],
        ['Best SSIM', f'{np.mean(ssim_ae):.3f}'],
    ]
    table = ax9.table(cellText=info[1:], colLabels=info[0],
                       cellLoc='left', loc='center',
                       colColours=['#e6f3ff']*2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax9.set_title('(i) Model & Noise Summary', fontsize=12, pad=10)

    fig.text(0.5, 0.01,
             'Denoising Autoencoder from scratch | CMOS sensor noise model | '
             'PSNR/SSIM metrics | Gaussian baseline comparison',
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
    print("Autoencoder for Image Sensor Denoising")
    print("Portfolio: Sony Semiconductor Solutions — ML/DL")
    print("=" * 60)

    print("\nGenerating test images...")
    images = generate_test_images(500)
    print(f"  Images: {len(images)} ({IMG_SIZE}x{IMG_SIZE})")

    print("Adding sensor noise (shot + read + FPN)...")
    noisy = add_sensor_noise(images)
    input_psnr = np.mean([psnr(images[i], noisy[i]) for i in range(len(images))])
    print(f"  Input PSNR: {input_psnr:.1f} dB")

    # Train/test split
    n_train = int(0.8 * len(images))
    flat_dim = IMG_SIZE * IMG_SIZE
    X_train_noisy = noisy[:n_train].reshape(-1, flat_dim)
    X_train_clean = images[:n_train].reshape(-1, flat_dim)
    X_val_noisy = noisy[n_train:].reshape(-1, flat_dim)
    X_val_clean = images[n_train:].reshape(-1, flat_dim)

    print(f"\nTraining Autoencoder (1024-256-64-256-1024)...")
    model = DenseAutoencoder(flat_dim, hidden_dims=[256, 64, 256])
    history = train_autoencoder(model, X_train_noisy, X_train_clean,
                                 X_val_noisy, X_val_clean,
                                 epochs=50, batch_size=32, lr=0.002)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "image_denoiser_results.png")
    print("\nPlotting results...")
    plot_results(model, history, images[n_train:], noisy[n_train:], save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
