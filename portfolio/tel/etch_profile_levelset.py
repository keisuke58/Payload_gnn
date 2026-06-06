#!/usr/bin/env python3
"""
Level-Set Etch Profile Simulator
==================================
Portfolio: Tokyo Electron (TEL) — Etching Equipment

Simulates surface evolution during semiconductor plasma etching
using the Level-Set method.

Physics:
- Hamilton-Jacobi equation: φ_t + V|∇φ| = 0
- Isotropic etch: uniform etch rate in all directions (chemical/radical)
- Anisotropic etch: directional etch (ion bombardment, vertical preferred)
- Combined: tunable isotropy ratio α ∈ [0, 1]
- Photoresist mask with etch selectivity

Numerics:
- Godunov upwind scheme for spatial gradients
- TVD Runge-Kutta 3rd order (SSP-RK3) for time integration
- PDE-based reinitialization (Sussman-Smereka-Osher method)

Author: Nishioka
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time

# ============================================================
# Grid parameters
# ============================================================
class EtchSimulator:
    def __init__(self, Lx=4.0, Ly=4.0, nx=200, ny=200):
        """
        Lx, Ly: domain size [μm]
        nx, ny: grid cells
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.x = np.linspace(-Lx/2, Lx/2, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def init_level_set(self, substrate_y, mask_specs):
        """
        Initialize level-set function.
        φ < 0: material (solid)
        φ > 0: void (gas/plasma)

        substrate_y: y-coordinate of substrate surface [μm]
        mask_specs: list of (x_start, x_end) for mask openings
        """
        # Start with φ = y - substrate_y (positive above surface)
        phi = self.Y - substrate_y

        # Add photoresist mask (negative phi = solid)
        mask_thickness = 0.5  # μm
        mask_top = substrate_y + mask_thickness

        for x_start, x_end in mask_specs:
            # Where mask exists (outside openings), set solid
            mask_region = (
                (self.Y >= substrate_y) &
                (self.Y <= mask_top) &
                ~((self.X >= x_start) & (self.X <= x_end))
            )
            phi[mask_region] = -np.abs(phi[mask_region]) - 0.01

        return phi

    def godunov_gradient(self, phi):
        """
        Godunov upwind scheme for |∇φ| used in etch velocity.
        For V > 0 (material removal): use upwind from solid side.
        """
        dx, dy = self.dx, self.dy

        # x-derivatives (forward and backward)
        Dxp = np.zeros_like(phi)
        Dxm = np.zeros_like(phi)
        Dxp[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dx
        Dxm[1:, :] = (phi[1:, :] - phi[:-1, :]) / dx

        # y-derivatives
        Dyp = np.zeros_like(phi)
        Dym = np.zeros_like(phi)
        Dyp[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dy
        Dym[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dy

        return Dxp, Dxm, Dyp, Dym

    def etch_speed(self, phi, alpha, etch_rate, etch_rate_mask=None):
        """
        Compute level-set velocity field.
        alpha: isotropy ratio (0=fully anisotropic, 1=fully isotropic)
        etch_rate: etch rate [μm/s]
        """
        Dxp, Dxm, Dyp, Dym = self.godunov_gradient(phi)

        # Godunov Hamiltonian for V > 0 (etching into material)
        # Anisotropic component: only vertical (y-direction downward)
        # Isotropic component: all directions equally

        # Upwind selection for positive speed (V > 0)
        Dxp_max = np.maximum(Dxp, 0)
        Dxm_min = np.minimum(Dxm, 0)
        Dyp_max = np.maximum(Dyp, 0)
        Dym_min = np.minimum(Dym, 0)

        # Isotropic: |∇φ| using Godunov
        grad_iso = np.sqrt(
            np.maximum(Dxp_max**2, Dxm_min**2) +
            np.maximum(Dyp_max**2, Dym_min**2)
        )

        # Anisotropic: only downward component (-y direction)
        # Speed proportional to cos(θ) where θ is angle from vertical
        grad_aniso = np.sqrt(
            0.01 * np.maximum(Dxp_max**2, Dxm_min**2) +  # Small lateral component
            np.maximum(Dyp_max**2, Dym_min**2)             # Full vertical
        )

        # Combined velocity
        V_mag = etch_rate * (alpha * grad_iso + (1 - alpha) * grad_aniso)

        # Reduce etch rate in mask region (selectivity)
        if etch_rate_mask is not None:
            # Simple: where phi was initially solid mask, reduce rate
            V_mag *= etch_rate_mask

        return V_mag

    def rk3_step(self, phi, dt, alpha, etch_rate, mask_selectivity):
        """SSP Runge-Kutta 3rd order time integration."""
        # Stage 1
        L1 = -self.etch_speed(phi, alpha, etch_rate, mask_selectivity)
        phi1 = phi + dt * L1

        # Stage 2
        L2 = -self.etch_speed(phi1, alpha, etch_rate, mask_selectivity)
        phi2 = 0.75 * phi + 0.25 * (phi1 + dt * L2)

        # Stage 3
        L3 = -self.etch_speed(phi2, alpha, etch_rate, mask_selectivity)
        phi_new = (1.0/3.0) * phi + (2.0/3.0) * (phi2 + dt * L3)

        return phi_new

    def reinitialize(self, phi, n_iter=5):
        """
        PDE-based reinitialization (Sussman-Smereka-Osher 1994).
        Solve: φ_τ + sign(φ₀)(|∇φ| - 1) = 0 to steady state.
        Restores φ to a signed distance function.
        """
        phi0 = phi.copy()
        dtau = 0.5 * min(self.dx, self.dy)

        for _ in range(n_iter):
            # Smoothed sign function
            eps = 1.5 * max(self.dx, self.dy)
            S = phi0 / np.sqrt(phi0**2 + eps**2)

            # Upwind derivatives based on sign
            Dxp, Dxm, Dyp, Dym = self.godunov_gradient(phi)

            # Godunov scheme for sign(φ₀)(|∇φ| - 1)
            # Where S > 0: use max(Dxm,0)² + min(Dxp,0)² etc.
            # Where S < 0: use max(Dxp,0)² + min(Dxm,0)² etc.

            grad_plus = np.sqrt(
                np.maximum(np.maximum(Dxm, 0)**2, np.minimum(Dxp, 0)**2) +
                np.maximum(np.maximum(Dym, 0)**2, np.minimum(Dyp, 0)**2)
            )
            grad_minus = np.sqrt(
                np.maximum(np.maximum(Dxp, 0)**2, np.minimum(Dxm, 0)**2) +
                np.maximum(np.maximum(Dyp, 0)**2, np.minimum(Dym, 0)**2)
            )

            phi -= dtau * (np.maximum(S, 0) * (grad_plus - 1) +
                           np.minimum(S, 0) * (grad_minus - 1))

        return phi

    def run(self, alpha, etch_rate, substrate_y, mask_openings,
            t_total, n_snapshots=5, mask_selectivity_ratio=0.05):
        """
        Run etch simulation.
        Returns: list of (time, phi) snapshots
        """
        phi = self.init_level_set(substrate_y, mask_openings)

        # Mask selectivity field: 1.0 in etched region, reduced in mask
        mask_sel = np.ones_like(phi)
        mask_thickness = 0.5
        for x_s, x_e in mask_openings:
            pass  # Openings are already 1.0
        # Reduce in mask (initial solid above substrate that's not in opening)
        mask_region = (
            (self.Y >= substrate_y) &
            (self.Y <= substrate_y + mask_thickness)
        )
        is_opening = np.zeros_like(phi, dtype=bool)
        for x_s, x_e in mask_openings:
            is_opening |= (self.X >= x_s) & (self.X <= x_e)
        mask_sel[mask_region & ~is_opening] = mask_selectivity_ratio

        # CFL condition
        dt = 0.4 * min(self.dx, self.dy) / etch_rate
        n_steps = int(t_total / dt)
        snapshot_interval = max(1, n_steps // n_snapshots)

        snapshots = [(0.0, phi.copy())]

        for step in range(n_steps):
            phi = self.rk3_step(phi, dt, alpha, etch_rate, mask_sel)

            # Reinitialize periodically
            if (step + 1) % 20 == 0:
                phi = self.reinitialize(phi, n_iter=3)

            if (step + 1) % snapshot_interval == 0:
                t = (step + 1) * dt
                snapshots.append((t, phi.copy()))

        return snapshots

# ============================================================
# Visualization
# ============================================================
def plot_etch_comparison(results_dict, sim, save_path):
    """
    Multi-panel figure comparing different etch scenarios.
    results_dict: {name: snapshots}
    """
    n_scenarios = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Level-Set Etch Profile Simulation — Semiconductor Etching (TEL)',
                 fontsize=15, fontweight='bold')

    colors_solid = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#FFE4C4']
    colors_mask = '#228B22'

    for idx, (name, snapshots) in enumerate(results_dict.items()):
        if idx >= 5:
            break
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Plot each snapshot as contour
        for i, (t, phi) in enumerate(snapshots):
            color = f'C{i}'
            ax.contour(sim.X, sim.Y, phi, levels=[0],
                      colors=[color], linewidths=1.5)

        # Final profile fill
        t_final, phi_final = snapshots[-1]
        ax.contourf(sim.X, sim.Y, phi_final, levels=[-100, 0],
                   colors=['#D2B48C'], alpha=0.4)
        ax.contour(sim.X, sim.Y, phi_final, levels=[0],
                  colors=['#8B0000'], linewidths=2)

        # Initial profile
        t0, phi0 = snapshots[0]
        ax.contour(sim.X, sim.Y, phi0, levels=[0],
                  colors=['gray'], linewidths=1, linestyles='--')

        ax.set_xlabel('x [μm]')
        ax.set_ylabel('y [μm]')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Add time labels
        for i, (t, _) in enumerate(snapshots):
            if i == 0:
                continue
            ax.text(0.98, 0.98 - i*0.06, f't={t:.2f}s',
                   transform=ax.transAxes, fontsize=8,
                   ha='right', va='top', color=f'C{i}')

    # (f) Cross-section comparison at final time
    ax6 = axes[1, 2]
    for name, snapshots in results_dict.items():
        _, phi_final = snapshots[-1]
        # Extract profile at x=0
        ix_center = sim.nx // 2
        profile = phi_final[ix_center, :]
        ax6.plot(sim.y, profile, linewidth=2, label=name)

    ax6.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax6.set_xlabel('y [μm]')
    ax6.set_ylabel('Level-Set φ')
    ax6.set_title('(f) Level-Set Profile at x=0 (Cross-Section)', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Level-Set Etch Profile Simulator")
    print("Portfolio: Tokyo Electron (TEL)")
    print("=" * 60)

    # Common parameters
    substrate_y = 2.5   # Substrate surface at y=2.5 μm
    etch_rate = 2.0     # μm/s
    t_etch = 0.5        # Total etch time [s]
    opening = [(-0.5, 0.5)]  # Single trench opening

    sim = EtchSimulator(Lx=4.0, Ly=4.0, nx=200, ny=200)
    results = {}

    # --- Scenario 1: Pure isotropic etch (wet etch) ---
    print("\n[1/5] Pure isotropic etch (α=1.0)...")
    results['(a) Isotropic (α=1.0)\nWet Etch'] = sim.run(
        alpha=1.0, etch_rate=etch_rate, substrate_y=substrate_y,
        mask_openings=opening, t_total=t_etch, n_snapshots=5
    )

    # --- Scenario 2: Pure anisotropic etch (ideal RIE) ---
    print("[2/5] Pure anisotropic etch (α=0.0)...")
    results['(b) Anisotropic (α=0.0)\nIdeal RIE'] = sim.run(
        alpha=0.0, etch_rate=etch_rate, substrate_y=substrate_y,
        mask_openings=opening, t_total=t_etch, n_snapshots=5
    )

    # --- Scenario 3: Combined etch ---
    print("[3/5] Combined etch (α=0.3)...")
    results['(c) Combined (α=0.3)\nTypical RIE'] = sim.run(
        alpha=0.3, etch_rate=etch_rate, substrate_y=substrate_y,
        mask_openings=opening, t_total=t_etch, n_snapshots=5
    )

    # --- Scenario 4: High-aspect-ratio trench (3D NAND) ---
    print("[4/5] High-aspect-ratio trench (3D NAND)...")
    sim_har = EtchSimulator(Lx=2.0, Ly=6.0, nx=100, ny=300)
    narrow_opening = [(-0.15, 0.15)]  # 0.3 μm wide trench
    results['(d) HAR Trench (3D NAND)\nAR > 10:1'] = sim_har.run(
        alpha=0.05, etch_rate=3.0, substrate_y=3.0,
        mask_openings=narrow_opening, t_total=0.6, n_snapshots=5
    )

    # --- Scenario 5: Multi-trench (varying pitch) ---
    print("[5/5] Multi-trench with varying pitch...")
    sim_multi = EtchSimulator(Lx=6.0, Ly=4.0, nx=300, ny=200)
    multi_openings = [
        (-2.5, -2.0),   # Pitch 1
        (-1.2, -0.7),   # Pitch 2
        (-0.1, 0.4),    # Pitch 3
        (1.0, 1.5),     # Pitch 4
        (2.0, 2.5),     # Pitch 5
    ]
    results['(e) Multi-Trench\nVarying Pitch'] = sim_multi.run(
        alpha=0.15, etch_rate=2.0, substrate_y=2.5,
        mask_openings=multi_openings, t_total=0.5, n_snapshots=5
    )

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "etch_profile_results.png")
    print("\nPlotting results...")

    # Use the base sim for single-trench scenarios
    # Create a unified plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Level-Set Etch Profile Simulation — Semiconductor Etching (TEL)',
                 fontsize=15, fontweight='bold')

    for idx, (name, snapshots) in enumerate(results.items()):
        if idx >= 5:
            break
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Determine which simulator was used
        if 'HAR' in name:
            s = sim_har
        elif 'Multi' in name:
            s = sim_multi
        else:
            s = sim

        # Plot each snapshot
        for i, (t, phi) in enumerate(snapshots):
            ax.contour(s.X, s.Y, phi, levels=[0],
                      colors=[f'C{i}'], linewidths=1.5)

        # Final profile fill
        t_final, phi_final = snapshots[-1]
        ax.contourf(s.X, s.Y, phi_final, levels=[-1000, 0],
                   colors=['#D2B48C'], alpha=0.5)
        ax.contour(s.X, s.Y, phi_final, levels=[0],
                  colors=['darkred'], linewidths=2.5)

        # Initial
        ax.contour(s.X, s.Y, snapshots[0][1], levels=[0],
                  colors=['gray'], linewidths=1, linestyles='--')

        ax.set_xlabel('x [μm]')
        ax.set_ylabel('y [μm]')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        for i, (t, _) in enumerate(snapshots):
            if i == 0:
                continue
            ax.text(0.98, 0.98 - i*0.05, f't={t:.2f}s',
                   transform=ax.transAxes, fontsize=8,
                   ha='right', va='top', color=f'C{i}')

    # (f) Undercut comparison
    ax6 = axes[1, 2]
    alphas = [0.0, 0.15, 0.3, 0.5, 1.0]
    for a in alphas:
        snaps = sim.run(alpha=a, etch_rate=2.0, substrate_y=substrate_y,
                       mask_openings=opening, t_total=0.4, n_snapshots=1)
        _, phi_f = snaps[-1]
        ax6.contour(sim.X, sim.Y, phi_f, levels=[0], linewidths=2,
                   colors=[plt.cm.plasma(a)])

    ax6.contour(sim.X, sim.Y, snaps[0][1], levels=[0],
               colors=['gray'], linewidths=1, linestyles='--')
    ax6.set_xlabel('x [μm]')
    ax6.set_ylabel('y [μm]')
    ax6.set_title('(f) Undercut vs Isotropy Ratio α\n(0.0, 0.15, 0.3, 0.5, 1.0)',
                 fontsize=11, fontweight='bold')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.2)

    # Colorbar for alpha
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax6, label='Isotropy ratio α', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
