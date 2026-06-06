#!/usr/bin/env python3
"""
2D FDTD Electromagnetic Wave Solver
=====================================
Portfolio: Lasertec Corporation — EUV Mask Inspection

Simulates electromagnetic wave propagation and scattering from
a photomask structure using the Finite-Difference Time-Domain method.

Physics:
- Maxwell's equations in 2D (TM mode: Ez, Hx, Hy)
- Yee grid: staggered spatial and temporal discretization
- CPML (Convolutional Perfectly Matched Layer) absorbing boundaries
- Materials: dielectric structures with complex permittivity
- Application: EUV/DUV mask defect scattering analysis

Numerics:
- Courant stability: dt ≤ dx / (c√2)  for 2D
- Spatial resolution: dx ≤ λ/20 (minimum)
- CPML: polynomial grading of conductivity

Author: Nishioka
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time

# ============================================================
# Physical constants
# ============================================================
c0 = 2.99792458e8   # Speed of light [m/s]
mu0 = 4e-7 * np.pi  # Permeability of free space [H/m]
eps0 = 1.0 / (mu0 * c0**2)  # Permittivity of free space [F/m]

# ============================================================
# FDTD Solver
# ============================================================
class FDTD2D:
    """2D FDTD solver for TM mode (Ez, Hx, Hy)."""

    def __init__(self, nx, ny, dx, dy, wavelength):
        """
        nx, ny: grid size
        dx, dy: cell size [m]
        wavelength: source wavelength [m]
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.wavelength = wavelength
        self.freq = c0 / wavelength

        # Courant condition
        self.dt = 0.99 / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # Fields
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

        # Material properties (relative permittivity and conductivity)
        self.eps_r = np.ones((nx, ny))
        self.sigma = np.zeros((nx, ny))

        # CPML
        self.pml_thickness = 15
        self._setup_cpml()

        # Time recording
        self.time_steps = 0

    def _setup_cpml(self):
        """Setup CPML absorbing boundary parameters."""
        d = self.pml_thickness
        nx, ny = self.nx, self.ny

        # Polynomial grading
        sigma_max = 0.8 * (3 + 1) / (self.dx * np.sqrt(mu0 / eps0))
        kappa_max = 1.0
        alpha_max = 0.05

        # PML profiles for x and y
        self.psi_Ezx1 = np.zeros((d, ny))  # left
        self.psi_Ezx2 = np.zeros((d, ny))  # right
        self.psi_Ezy1 = np.zeros((nx, d))  # bottom
        self.psi_Ezy2 = np.zeros((nx, d))  # top

        self.psi_Hyx1 = np.zeros((d, ny))
        self.psi_Hyx2 = np.zeros((d, ny))
        self.psi_Hxy1 = np.zeros((nx, d))
        self.psi_Hxy2 = np.zeros((nx, d))

        # Grading arrays
        self.be_x = np.zeros(d)
        self.ce_x = np.zeros(d)
        self.bh_x = np.zeros(d)
        self.ch_x = np.zeros(d)
        self.be_y = np.zeros(d)
        self.ce_y = np.zeros(d)
        self.bh_y = np.zeros(d)
        self.ch_y = np.zeros(d)

        for i in range(d):
            # E-field positions (integer + 0.5)
            pos_e = (d - i - 0.5) / d
            sigma_e = sigma_max * pos_e**3
            kappa_e = 1 + (kappa_max - 1) * pos_e**3
            alpha_e = alpha_max * (1 - pos_e)
            self.be_x[i] = np.exp(-(sigma_e/kappa_e + alpha_e) * self.dt / eps0)
            if abs(sigma_e) > 1e-20:
                self.ce_x[i] = sigma_e / (sigma_e * kappa_e + kappa_e**2 * alpha_e) * (self.be_x[i] - 1)
            self.be_y[i] = self.be_x[i]
            self.ce_y[i] = self.ce_x[i]

            # H-field positions (integer)
            pos_h = (d - i) / d
            sigma_h = sigma_max * pos_h**3
            kappa_h = 1 + (kappa_max - 1) * pos_h**3
            alpha_h = alpha_max * (1 - pos_h)
            self.bh_x[i] = np.exp(-(sigma_h/kappa_h + alpha_h) * self.dt / eps0)
            if abs(sigma_h) > 1e-20:
                self.ch_x[i] = sigma_h / (sigma_h * kappa_h + kappa_h**2 * alpha_h) * (self.bh_x[i] - 1)
            self.bh_y[i] = self.bh_x[i]
            self.ch_y[i] = self.ch_x[i]

    def add_structure(self, x_range, y_range, eps_r, sigma=0.0):
        """Add a dielectric/absorbing structure."""
        x1, x2 = x_range
        y1, y2 = y_range
        self.eps_r[x1:x2, y1:y2] = eps_r
        self.sigma[x1:x2, y1:y2] = sigma

    def step(self, source_func=None, source_pos=None):
        """Advance one FDTD time step."""
        nx, ny = self.nx, self.ny
        dt = self.dt
        dx, dy = self.dx, self.dy
        d = self.pml_thickness

        # Update H-fields: Hx, Hy
        # Hx(n+1/2) = Hx(n-1/2) - dt/(μ₀ dy) * (Ez[i,j+1] - Ez[i,j])
        self.Hx[:, :-1] -= dt / (mu0 * dy) * (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Hy(n+1/2) = Hy(n-1/2) + dt/(μ₀ dx) * (Ez[i+1,j] - Ez[i,j])
        self.Hy[:-1, :] += dt / (mu0 * dx) * (self.Ez[1:, :] - self.Ez[:-1, :])

        # CPML for H (simplified — apply corrections at boundaries)
        # Left x-boundary
        for i in range(d):
            idx = d - 1 - i
            self.psi_Hyx1[i, :] = (self.bh_x[idx] * self.psi_Hyx1[i, :] +
                                    self.ch_x[idx] * (self.Ez[i+1, :] - self.Ez[i, :]) / dx)
            self.Hy[i, :] += dt / mu0 * self.psi_Hyx1[i, :]

        # Right x-boundary
        for i in range(d):
            idx = i
            ii = nx - d + i - 1
            if ii >= 0 and ii < nx - 1:
                self.psi_Hyx2[i, :] = (self.bh_x[idx] * self.psi_Hyx2[i, :] +
                                        self.ch_x[idx] * (self.Ez[ii+1, :] - self.Ez[ii, :]) / dx)
                self.Hy[ii, :] += dt / mu0 * self.psi_Hyx2[i, :]

        # Bottom y-boundary
        for j in range(d):
            jdx = d - 1 - j
            self.psi_Hxy1[:, j] = (self.bh_y[jdx] * self.psi_Hxy1[:, j] +
                                    self.ch_y[jdx] * (self.Ez[:, j+1] - self.Ez[:, j]) / dy)
            self.Hx[:, j] -= dt / mu0 * self.psi_Hxy1[:, j]

        # Top y-boundary
        for j in range(d):
            jdx = j
            jj = ny - d + j - 1
            if jj >= 0 and jj < ny - 1:
                self.psi_Hxy2[:, j] = (self.bh_y[jdx] * self.psi_Hxy2[:, j] +
                                        self.ch_y[jdx] * (self.Ez[:, jj+1] - self.Ez[:, jj]) / dy)
                self.Hx[:, jj] -= dt / mu0 * self.psi_Hxy2[:, j]

        # Update E-field: Ez
        # Ez(n+1) = Ca * Ez(n) + Cb * (dHy/dx - dHx/dy)
        Ca = (1 - self.sigma * dt / (2 * eps0 * self.eps_r)) / \
             (1 + self.sigma * dt / (2 * eps0 * self.eps_r))
        Cb = (dt / (eps0 * self.eps_r)) / \
             (1 + self.sigma * dt / (2 * eps0 * self.eps_r))

        dHy_dx = np.zeros((nx, ny))
        dHx_dy = np.zeros((nx, ny))
        dHy_dx[1:, :] = (self.Hy[1:, :] - self.Hy[:-1, :]) / dx
        dHx_dy[:, 1:] = (self.Hx[:, 1:] - self.Hx[:, :-1]) / dy

        self.Ez = Ca * self.Ez + Cb * (dHy_dx - dHx_dy)

        # CPML for E (simplified)
        for i in range(d):
            idx = d - 1 - i
            if i + 1 < nx:
                self.psi_Ezx1[i, :] = (self.be_x[idx] * self.psi_Ezx1[i, :] +
                                        self.ce_x[idx] * (self.Hy[i, :] - self.Hy[max(i-1,0), :]) / dx)
                self.Ez[i, :] += Cb[i, :] * self.psi_Ezx1[i, :]

        for i in range(d):
            ii = nx - d + i
            if 0 < ii < nx:
                self.psi_Ezx2[i, :] = (self.be_x[i] * self.psi_Ezx2[i, :] +
                                        self.ce_x[i] * (self.Hy[ii, :] - self.Hy[ii-1, :]) / dx)
                self.Ez[ii, :] += Cb[ii, :] * self.psi_Ezx2[i, :]

        for j in range(d):
            jdx = d - 1 - j
            if j + 1 < ny:
                self.psi_Ezy1[:, j] = (self.be_y[jdx] * self.psi_Ezy1[:, j] +
                                        self.ce_y[jdx] * (self.Hx[:, j] - self.Hx[:, max(j-1,0)]) / dy)
                self.Ez[:, j] -= Cb[:, j] * self.psi_Ezy1[:, j]

        for j in range(d):
            jj = ny - d + j
            if 0 < jj < ny:
                self.psi_Ezy2[:, j] = (self.be_y[j] * self.psi_Ezy2[:, j] +
                                        self.ce_y[j] * (self.Hx[:, jj] - self.Hx[:, jj-1]) / dy)
                self.Ez[:, jj] -= Cb[:, jj] * self.psi_Ezy2[:, j]

        # Add source
        if source_func is not None and source_pos is not None:
            sx, sy = source_pos
            self.Ez[sx, sy] += source_func(self.time_steps * dt)

        self.time_steps += 1

    def add_line_source(self, y_pos, source_func):
        """Add a line source across x at given y position."""
        t = self.time_steps * self.dt
        self.Ez[:, y_pos] += source_func(t)

# ============================================================
# Source functions
# ============================================================
def gaussian_pulse(t, t0, spread, freq):
    """Gaussian-modulated sinusoidal pulse."""
    return np.exp(-((t - t0) / spread)**2) * np.sin(2 * np.pi * freq * t)

def cw_source(t, freq, ramp_time=None):
    """Continuous wave with optional smooth ramp-up."""
    if ramp_time and t < ramp_time:
        envelope = 0.5 * (1 - np.cos(np.pi * t / ramp_time))
    else:
        envelope = 1.0
    return envelope * np.sin(2 * np.pi * freq * t)

# ============================================================
# Simulation setup: Photomask with line/space pattern + defect
# ============================================================
def setup_mask_simulation():
    """
    Simulate DUV light scattering from a photomask with
    line/space pattern and a defect (missing line).

    Using scaled wavelength (193 nm DUV → scaled for visualization).
    """
    # Wavelength and grid
    wavelength = 193e-9  # DUV wavelength [m]
    dx = wavelength / 25  # ~7.7 nm resolution
    dy = dx
    nx = 300
    ny = 400

    fdtd = FDTD2D(nx, ny, dx, dy, wavelength)

    # Build mask structure
    # Substrate: quartz (SiO2), eps_r ≈ 2.13 at 193 nm
    substrate_y = ny // 2
    fdtd.add_structure((0, nx), (0, substrate_y), eps_r=2.13)

    # Chrome absorber lines (eps_r ≈ -5 + 18j at 193 nm → high absorption)
    # Simplified: use high real eps_r + high conductivity
    line_width = int(0.15e-6 / dx)   # ~150 nm line width (≈19 cells)
    space_width = int(0.15e-6 / dx)  # ~150 nm space width
    absorber_height = int(0.07e-6 / dy)  # ~70 nm chrome thickness (≈9 cells)
    pitch = line_width + space_width

    # Place lines
    n_lines = nx // pitch
    for i in range(n_lines):
        x_start = i * pitch
        x_end = x_start + line_width

        # Skip one line to create a defect
        if i == n_lines // 2:
            continue  # DEFECT: missing absorber line

        if x_end <= nx:
            fdtd.add_structure(
                (x_start, x_end),
                (substrate_y, substrate_y + absorber_height),
                eps_r=3.0, sigma=5e7  # Simplified chrome properties
            )

    return fdtd, substrate_y, wavelength, line_width, space_width

# ============================================================
# Main simulation and visualization
# ============================================================
if __name__ == "__main__":
    t0_wall = time.time()
    print("=" * 60)
    print("2D FDTD Electromagnetic Wave Solver")
    print("Portfolio: Lasertec Corporation")
    print("=" * 60)

    # Setup
    print("\nSetting up photomask simulation (DUV 193 nm)...")
    fdtd, substrate_y, wavelength, lw, sw = setup_mask_simulation()
    freq = c0 / wavelength

    print(f"  Grid: {fdtd.nx} x {fdtd.ny}")
    print(f"  Cell size: {fdtd.dx*1e9:.1f} nm")
    print(f"  Wavelength: {wavelength*1e9:.0f} nm")
    print(f"  dt: {fdtd.dt:.3e} s")
    print(f"  Courant number: {c0 * fdtd.dt * np.sqrt(1/fdtd.dx**2 + 1/fdtd.dy**2):.3f}")

    # Source parameters
    source_y = fdtd.ny - 40  # Near top boundary
    t0_pulse = 3.0 / freq     # Pulse center time
    spread = 1.5 / freq        # Pulse width

    # Recording point
    rec_x, rec_y = fdtd.nx // 2, fdtd.ny - 60
    rec_x2, rec_y2 = fdtd.nx // 2, substrate_y - 20  # Below mask

    # Run simulation
    n_total_steps = 800
    snapshots = []
    snapshot_steps = [100, 250, 400, 600]
    Ez_rec1 = []
    Ez_rec2 = []
    time_rec = []

    print(f"\nRunning {n_total_steps} time steps...")
    for step in range(n_total_steps):
        t = step * fdtd.dt

        # Plane wave source (line source at top)
        source_val = gaussian_pulse(t, t0_pulse, spread, freq)
        fdtd.Ez[fdtd.pml_thickness:-fdtd.pml_thickness, source_y] += source_val

        fdtd.step()

        # Record
        Ez_rec1.append(fdtd.Ez[rec_x, rec_y])
        Ez_rec2.append(fdtd.Ez[rec_x2, rec_y2])
        time_rec.append(t)

        if step + 1 in snapshot_steps:
            snapshots.append((step + 1, fdtd.Ez.copy()))

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1}/{n_total_steps}")

    print("Simulation complete.")

    # Compute spectrum via FFT
    print("\nComputing spectrum...")
    Ez_arr1 = np.array(Ez_rec1)
    Ez_arr2 = np.array(Ez_rec2)
    t_arr = np.array(time_rec)

    n_fft = len(Ez_arr1)
    freq_axis = np.fft.rfftfreq(n_fft, d=fdtd.dt)
    spectrum_ref = np.abs(np.fft.rfft(Ez_arr1))
    spectrum_trans = np.abs(np.fft.rfft(Ez_arr2))

    # Convert to wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        wl_axis = np.where(freq_axis > 0, c0 / freq_axis * 1e9, np.inf)  # [nm]

    # ============================================================
    # Visualization
    # ============================================================
    print("Plotting results...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('2D FDTD Electromagnetic Wave Simulation — Photomask Inspection (Lasertec)',
                 fontsize=15, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Convert coordinates to nm
    x_nm = np.arange(fdtd.nx) * fdtd.dx * 1e9
    y_nm = np.arange(fdtd.ny) * fdtd.dy * 1e9

    # (a) Structure (permittivity map)
    ax1 = fig.add_subplot(gs[0, 0])
    eps_plot = fdtd.eps_r.T
    pcm = ax1.pcolormesh(x_nm, y_nm, eps_plot, cmap='gray_r', shading='auto')
    plt.colorbar(pcm, ax=ax1, label='εr')
    ax1.set_xlabel('x [nm]')
    ax1.set_ylabel('y [nm]')
    ax1.set_title('(a) Mask Structure (Permittivity Map)')

    # Mark defect
    defect_x = fdtd.nx // 2 * fdtd.dx * 1e9
    ax1.annotate('Defect\n(missing line)', xy=(defect_x, substrate_y * fdtd.dy * 1e9),
                fontsize=9, ha='center', color='red',
                xytext=(defect_x, (substrate_y + 40) * fdtd.dy * 1e9),
                arrowprops=dict(arrowstyle='->', color='red'))

    # (b, c) Field snapshots
    for idx, (si, snap_idx) in enumerate([(1, -2), (2, -1)]):
        ax = fig.add_subplot(gs[0, si])
        step_n, Ez_snap = snapshots[snap_idx]
        vmax = np.abs(Ez_snap).max() * 0.5
        if vmax < 1e-20:
            vmax = 1.0
        pcm = ax.pcolormesh(x_nm, y_nm, Ez_snap.T,
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(pcm, ax=ax, label='Ez [V/m]')
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        t_ps = step_n * fdtd.dt * 1e15
        label = chr(ord('b') + idx)
        ax.set_title(f'({label}) Ez Field (step {step_n}, t={t_ps:.1f} fs)')

        # Overlay structure outline
        ax.contour(x_nm, y_nm, fdtd.eps_r.T, levels=[1.5],
                  colors='yellow', linewidths=0.5, alpha=0.5)

    # (d) Time signal at recording point
    ax4 = fig.add_subplot(gs[1, 0])
    t_fs = np.array(time_rec) * 1e15
    ax4.plot(t_fs, Ez_arr1, 'b-', linewidth=1, label='Reflected (above mask)')
    ax4.plot(t_fs, Ez_arr2, 'r-', linewidth=1, label='Transmitted (below mask)')
    ax4.set_xlabel('Time [fs]')
    ax4.set_ylabel('Ez [V/m]')
    ax4.set_title('(d) Time Signal at Recording Points')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # (e) Spectrum
    ax5 = fig.add_subplot(gs[1, 1])
    valid = (wl_axis > 100) & (wl_axis < 500)
    if spectrum_ref[valid].max() > 0:
        ax5.plot(wl_axis[valid], spectrum_ref[valid] / spectrum_ref[valid].max(),
                'b-', linewidth=2, label='Reflected')
    if spectrum_trans[valid].max() > 0:
        ax5.plot(wl_axis[valid], spectrum_trans[valid] / spectrum_trans[valid].max(),
                'r-', linewidth=2, label='Transmitted')
    ax5.axvline(193, color='green', linestyle='--', alpha=0.5, label='DUV 193 nm')
    ax5.set_xlabel('Wavelength [nm]')
    ax5.set_ylabel('Normalized Amplitude')
    ax5.set_title('(e) Spectral Response')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(100, 400)

    # (f) Scattered field cross-section at observation plane
    ax6 = fig.add_subplot(gs[1, 2])
    # Extract Ez along x at reflected observation plane
    obs_y = fdtd.ny - 60
    if len(snapshots) > 0:
        _, Ez_final = snapshots[-1]
        Ez_line = Ez_final[:, obs_y]
        ax6.plot(x_nm, np.abs(Ez_line), 'b-', linewidth=2)
        ax6.set_xlabel('x [nm]')
        ax6.set_ylabel('|Ez| [V/m]')
        ax6.set_title(f'(f) Scattered Field at Observation Plane (y={obs_y * fdtd.dy * 1e9:.0f} nm)')
        ax6.grid(True, alpha=0.3)

        # Mark defect position
        ax6.axvline(defect_x, color='red', linestyle='--', alpha=0.5, label='Defect position')
        ax6.legend(fontsize=9)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "fdtd_results.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    elapsed = time.time() - t0_wall
    print(f"\nTotal elapsed: {elapsed:.1f} s")
