#!/usr/bin/env python3
"""
Gradient-Based Inverse Lithography / OPC Optimization
=======================================================
Portfolio: ASML Japan — ML/Optimization

Optimizes photomask patterns to achieve target wafer prints
using gradient-based inverse lithography optimization.

Optimization/ML Concepts:
  - Hopkins imaging model (partially coherent lithography)
  - Gradient descent with Adam optimizer
  - Sigmoid mask parametrization (continuous relaxation)
  - L1/TV regularization for mask manufacturability
  - ILT (Inverse Lithography Technology) pipeline

Domain:
  - Computational lithography for EUV/DUV
  - OPC (Optical Proximity Correction)
  - Mask optimization for advanced nodes

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
# Lithography Imaging Model
# ============================================================
WAVELENGTH = 193e-3   # DUV 193nm [um]
NA = 1.35             # Numerical aperture (immersion)
SIGMA = 0.7           # Partial coherence factor
PIXEL_SIZE = 5e-3     # 5 nm pixel [um]
GRID_SIZE = 128       # 128 x 128 grid = 640nm x 640nm

def create_pupil_function(grid_size, pixel_size, wavelength, na):
    """Create circular pupil function in frequency domain."""
    freq_x = np.fft.fftfreq(grid_size, d=pixel_size)
    fx, fy = np.meshgrid(freq_x, freq_x)
    cutoff = na / wavelength
    pupil = np.sqrt(fx**2 + fy**2) <= cutoff
    return pupil.astype(float)

def create_source_points(sigma, na, wavelength, n_points=9):
    """Create source points for partially coherent imaging (Abbe method)."""
    max_freq = sigma * na / wavelength
    # 3x3 grid of source points
    n_side = int(np.sqrt(n_points))
    sx = np.linspace(-max_freq, max_freq, n_side)
    sources = []
    for fx in sx:
        for fy in sx:
            if np.sqrt(fx**2 + fy**2) <= max_freq:
                sources.append((fx, fy))
    return sources

def aerial_image(mask, pupil, sources, pixel_size):
    """Compute aerial image using Hopkins/Abbe partially coherent model."""
    grid_size = mask.shape[0]
    freq_x = np.fft.fftfreq(grid_size, d=pixel_size)
    fx, fy = np.meshgrid(freq_x, freq_x)

    intensity = np.zeros((grid_size, grid_size))
    mask_fft = np.fft.fft2(mask)

    for src_fx, src_fy in sources:
        # Shifted pupil for this source point
        shifted_r = np.sqrt((fx - src_fx)**2 + (fy - src_fy)**2)
        cutoff = NA / WAVELENGTH
        H = (shifted_r <= cutoff).astype(float)

        # Filtered mask spectrum
        filtered = mask_fft * H
        field = np.fft.ifft2(filtered)
        intensity += np.abs(field)**2

    intensity /= len(sources)
    return intensity

def threshold_resist(aerial, threshold=0.3):
    """Simple threshold resist model."""
    return (aerial > threshold).astype(float)

# ============================================================
# Target Pattern
# ============================================================
def create_target_pattern(grid_size=GRID_SIZE):
    """Create target wafer pattern: L/S with contact holes."""
    target = np.zeros((grid_size, grid_size))

    # Line/space pattern (3 lines, pitch = 32 pixels = 160nm)
    for cx in [32, 64, 96]:
        half_w = 6  # ~30nm half-width
        target[:, max(0, cx-half_w):min(grid_size, cx+half_w)] = 1.0

    # Contact holes
    for (cx, cy) in [(48, 32), (48, 64), (48, 96), (80, 48), (80, 80)]:
        r = 5  # ~25nm radius
        yy, xx = np.ogrid[:grid_size, :grid_size]
        mask_hole = (xx - cx)**2 + (yy - cy)**2 <= r**2
        target[mask_hole] = 1.0

    return target

# ============================================================
# Inverse Lithography (ILT) with Adam Optimizer
# ============================================================
class InverseLithography:
    """Gradient-based mask optimization (ILT)."""

    def __init__(self, target, pupil, sources, pixel_size,
                 threshold=0.3, reg_tv=0.001, reg_l1=0.0005):
        self.target = target
        self.pupil = pupil
        self.sources = sources
        self.pixel_size = pixel_size
        self.threshold = threshold
        self.reg_tv = reg_tv
        self.reg_l1 = reg_l1
        self.grid_size = target.shape[0]

        # Initialize mask as sigmoid parametrization
        self.z = np.zeros_like(target)  # Latent variable
        # Initialize z from target
        self.z = np.where(target > 0.5, 2.0, -2.0)

    def sigmoid(self, z, beta=10.0):
        """Smooth sigmoid for differentiable mask."""
        return 1.0 / (1.0 + np.exp(-beta * np.clip(z, -10, 10)))

    def sigmoid_grad(self, z, beta=10.0):
        """Gradient of sigmoid."""
        s = self.sigmoid(z, beta)
        return beta * s * (1 - s)

    def forward(self, z, beta=10.0):
        """Compute aerial image from latent mask variable."""
        mask = self.sigmoid(z, beta)
        ai = aerial_image(mask, self.pupil, self.sources, self.pixel_size)
        return mask, ai

    def compute_loss(self, ai, mask):
        """Compute loss: pattern error + TV + L1 regularization."""
        # Pattern error (MSE between aerial image and target)
        pattern_loss = np.mean((ai - self.target)**2)

        # Total variation (TV) for mask smoothness / manufacturability
        tv_x = np.mean(np.abs(np.diff(mask, axis=1)))
        tv_y = np.mean(np.abs(np.diff(mask, axis=0)))
        tv_loss = self.reg_tv * (tv_x + tv_y)

        # L1 for mask simplicity
        l1_loss = self.reg_l1 * np.mean(np.abs(mask - 0.5))

        return pattern_loss + tv_loss + l1_loss, pattern_loss, tv_loss

    def compute_gradient(self, z, beta=10.0):
        """Compute gradient of loss w.r.t. z (backprop through imaging model)."""
        mask = self.sigmoid(z, beta)
        ai = aerial_image(mask, self.pupil, self.sources, self.pixel_size)

        # d(loss)/d(ai) = 2 * (ai - target) / N
        N = self.grid_size**2
        d_ai = 2 * (ai - self.target) / N

        # d(ai)/d(mask) via adjoint method (approximate)
        # For each source point: intensity = |IFFT(FFT(mask)*H)|^2
        # d(intensity)/d(mask) = 2 * Re(IFFT(FFT(d_ai) * |H|^2 * conj(FFT(mask))))
        d_mask = np.zeros_like(mask)
        mask_fft = np.fft.fft2(mask)
        d_ai_fft = np.fft.fft2(d_ai)
        freq_x = np.fft.fftfreq(self.grid_size, d=self.pixel_size)
        fx, fy = np.meshgrid(freq_x, freq_x)

        for src_fx, src_fy in self.sources:
            shifted_r = np.sqrt((fx - src_fx)**2 + (fy - src_fy)**2)
            cutoff = NA / WAVELENGTH
            H = (shifted_r <= cutoff).astype(float)

            filtered = mask_fft * H
            field = np.fft.ifft2(filtered)

            # Adjoint: d_mask += 2*Re(IFFT(H * FFT(field * d_ai)))
            # Simplified gradient approximation
            d_field = 2 * field * np.fft.ifft2(d_ai_fft * H)
            d_mask += np.real(np.fft.ifft2(np.fft.fft2(np.real(d_field)) * H))

        d_mask /= len(self.sources)

        # Chain rule: d(loss)/d(z) = d(loss)/d(mask) * d(mask)/d(z)
        d_z = d_mask * self.sigmoid_grad(z, beta)

        return d_z, ai

    def optimize(self, n_iter=200, lr=0.05):
        """Adam optimizer for mask optimization."""
        m = np.zeros_like(self.z)
        v = np.zeros_like(self.z)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        history = {'loss': [], 'pattern_err': [], 'epe': []}
        beta_schedule = np.linspace(5, 20, n_iter)  # Sigmoid sharpening

        for it in range(n_iter):
            beta = beta_schedule[it]
            grad, ai = self.compute_gradient(self.z, beta)
            mask = self.sigmoid(self.z, beta)

            loss, pat_err, tv = self.compute_loss(ai, mask)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**(it+1))
            v_hat = v / (1 - beta2**(it+1))
            self.z -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Edge Placement Error
            print_ai = threshold_resist(ai)
            epe = np.mean(np.abs(print_ai - self.target)) * 100

            history['loss'].append(loss)
            history['pattern_err'].append(pat_err)
            history['epe'].append(epe)

            if (it + 1) % 50 == 0:
                print(f"  Iter {it+1}: loss={loss:.6f} EPE={epe:.2f}%")

        final_mask = self.sigmoid(self.z, beta_schedule[-1])
        final_ai = aerial_image(final_mask, self.pupil, self.sources, self.pixel_size)
        return final_mask, final_ai, history

# ============================================================
# Visualization
# ============================================================
def plot_results(target, init_mask, init_ai, opt_mask, opt_ai, history, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Inverse Lithography / OPC Optimization (ASML)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    extent = [0, GRID_SIZE*PIXEL_SIZE*1000, 0, GRID_SIZE*PIXEL_SIZE*1000]  # nm

    # (a) Target pattern
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target, cmap='gray', origin='lower', extent=extent)
    ax1.set_title('(a) Target Wafer Pattern')
    ax1.set_xlabel('x [nm]')
    ax1.set_ylabel('y [nm]')

    # (b) Initial mask (= target) and print
    ax2 = fig.add_subplot(gs[0, 1])
    init_print = threshold_resist(init_ai)
    ax2.contour(init_print, levels=[0.5], colors='red', linewidths=1,
               extent=extent)
    ax2.contour(target, levels=[0.5], colors='blue', linewidths=1,
               linestyles='--', extent=extent)
    ax2.imshow(init_ai, cmap='hot', origin='lower', alpha=0.5, extent=extent)
    ax2.set_title('(b) Before OPC (blue=target, red=print)')
    ax2.set_xlabel('x [nm]')
    ax2.set_ylabel('y [nm]')

    # (c) Optimized mask and print
    ax3 = fig.add_subplot(gs[0, 2])
    opt_print = threshold_resist(opt_ai)
    ax3.contour(opt_print, levels=[0.5], colors='red', linewidths=1,
               extent=extent)
    ax3.contour(target, levels=[0.5], colors='blue', linewidths=1,
               linestyles='--', extent=extent)
    ax3.imshow(opt_ai, cmap='hot', origin='lower', alpha=0.5, extent=extent)
    ax3.set_title('(c) After ILT/OPC (blue=target, red=print)')
    ax3.set_xlabel('x [nm]')
    ax3.set_ylabel('y [nm]')

    # (d) Loss convergence
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(history['loss'], 'b-', linewidth=2, label='Total Loss')
    ax4.semilogy(history['pattern_err'], 'r--', linewidth=1.5, label='Pattern Error')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss')
    ax4.set_title('(d) Optimization Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # (e) EPE improvement
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['epe'], 'g-', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Edge Placement Error [%]')
    ax5.set_title('(e) EPE Convergence')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(history['epe'][-1], color='red', linestyle=':', alpha=0.5,
               label=f'Final: {history["epe"][-1]:.2f}%')
    ax5.legend()

    # (f) Optimized mask detail
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(opt_mask, cmap='gray', origin='lower', extent=extent)
    ax6.set_title('(f) Optimized Mask (with OPC features)')
    ax6.set_xlabel('x [nm]')
    ax6.set_ylabel('y [nm]')

    # Compute before/after EPE
    init_epe = np.mean(np.abs(threshold_resist(init_ai) - target)) * 100
    final_epe = history['epe'][-1]

    fig.text(0.5, 0.01,
             f'ILT with Adam optimizer | Hopkins imaging model | '
             f'Sigmoid mask parametrization | '
             f'EPE: {init_epe:.1f}% -> {final_epe:.1f}% | '
             f'lambda={WAVELENGTH*1000:.0f}nm, NA={NA}',
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
    print("Inverse Lithography / OPC Optimization")
    print("Portfolio: ASML Japan — Computational Lithography")
    print("=" * 60)

    print(f"\n  Wavelength: {WAVELENGTH*1000:.0f}nm, NA: {NA}, sigma: {SIGMA}")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, pixel: {PIXEL_SIZE*1000:.0f}nm")

    # Setup optics
    print("\nSetting up imaging model...")
    pupil = create_pupil_function(GRID_SIZE, PIXEL_SIZE, WAVELENGTH, NA)
    sources = create_source_points(SIGMA, NA, WAVELENGTH, n_points=9)
    print(f"  Source points: {len(sources)}")

    # Target pattern
    target = create_target_pattern()

    # Initial (no OPC): use target as mask
    init_mask = target.copy()
    init_ai = aerial_image(init_mask, pupil, sources, PIXEL_SIZE)
    init_epe = np.mean(np.abs(threshold_resist(init_ai) - target)) * 100
    print(f"  Before OPC EPE: {init_epe:.2f}%")

    # ILT optimization
    print("\nRunning ILT optimization (200 iterations)...")
    ilt = InverseLithography(target, pupil, sources, PIXEL_SIZE)
    opt_mask, opt_ai, history = ilt.optimize(n_iter=200, lr=0.05)
    final_epe = history['epe'][-1]
    print(f"  After ILT EPE: {final_epe:.2f}%")
    print(f"  EPE reduction: {(1-final_epe/init_epe)*100:.1f}%")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "opc_inverse_results.png")
    print("\nPlotting results...")
    plot_results(target, init_mask, init_ai, opt_mask, opt_ai, history, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
