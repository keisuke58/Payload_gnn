#!/usr/bin/env python3
"""
2D Electrostatic PIC-MCC Plasma Simulator
==========================================
Portfolio: Tokyo Electron (TEL) — Plasma Etching Equipment

Simulates an Argon CCP (Capacitively Coupled Plasma) discharge
between parallel-plate electrodes using:
- Particle-In-Cell (PIC) with Cloud-In-Cell (CIC) weighting
- Monte Carlo Collisions (MCC) for e-Ar interactions
- FFT-based Poisson solver for electrostatic potential
- Boris pusher for particle motion

Physics:
- Electrons and Ar+ ions in 2D (x,y) with velocity in 3D (vx,vy,vz)
- Electrode gap: parallel plates (top=RF, bottom=grounded)
- Collisions: elastic, excitation (11.55 eV), ionization (15.76 eV)
- Secondary electron emission (SEE) from ion impact on electrode (γ=0.1)

Author: Nishioka
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time

# ============================================================
# Physical constants (SI)
# ============================================================
e_charge = 1.602176634e-19     # Elementary charge [C]
m_e = 9.1093837015e-31         # Electron mass [kg]
m_Ar = 39.948 * 1.66053906660e-27  # Argon mass [kg]
eps0 = 8.854187817e-12         # Vacuum permittivity [F/m]
k_B = 1.380649e-23             # Boltzmann constant [J/K]

# ============================================================
# Simulation parameters
# ============================================================
# Domain: 2D box (x: periodic, y: electrodes at y=0 and y=Ly)
Lx = 0.02       # Domain width [m] (2 cm)
Ly = 0.02       # Electrode gap [m] (2 cm)
nx = 64          # Grid cells in x
ny = 64          # Grid cells in y
dx = Lx / nx
dy = Ly / ny

# Plasma parameters
n0 = 1e15        # Initial plasma density [m⁻³]
T_e = 3.0        # Electron temperature [eV]
T_i = 0.026      # Ion temperature [eV] (~300 K)
p_gas = 5.0      # Gas pressure [Pa] (37.5 mTorr)
T_gas = 300.0    # Gas temperature [K]
n_gas = p_gas / (k_B * T_gas)  # Neutral gas density [m⁻³]

# RF drive
V_rf = 200.0     # RF voltage amplitude [V]
f_rf = 13.56e6   # RF frequency [Hz]

# Particle parameters
n_ppc = 20       # Particles per cell (initial)
weight = n0 * dx * dy / n_ppc  # Superparticle weight

# Time stepping
dt = 1.0 / f_rf / 100   # ~100 steps per RF cycle
n_steps = 500            # Total steps (~5 RF cycles)

# SEE coefficient
gamma_see = 0.1

# ============================================================
# Cross-section data for e-Ar (simplified analytical fits)
# ============================================================
def sigma_elastic(energy_eV):
    """Elastic scattering cross-section for e-Ar [m²].
    Fit based on Hayashi (1983) data."""
    if energy_eV < 0.01:
        return 1e-20
    E = energy_eV
    # Ramsauer minimum near 0.3 eV, peak near 10 eV
    sigma = 6e-20 * E / (E**2 + 1.0) + 1e-20 * np.exp(-((E - 0.3)/0.2)**2) * 0.5
    return max(sigma, 1e-22)

def sigma_excitation(energy_eV):
    """Excitation cross-section for e-Ar (threshold 11.55 eV) [m²]."""
    E_th = 11.55
    if energy_eV < E_th:
        return 0.0
    E = energy_eV - E_th
    return 5e-21 * E / (E + 3.0)

def sigma_ionization(energy_eV):
    """Ionization cross-section for e-Ar (threshold 15.76 eV) [m²].
    Based on BEB (Binary Encounter Bethe) model simplified."""
    E_th = 15.76
    if energy_eV < E_th:
        return 0.0
    u = energy_eV / E_th
    return 4e-20 * (1 - 1/u) * np.log(u) / (u * E_th)

def total_cross_section(energy_eV):
    """Total cross-section for null-collision method."""
    return sigma_elastic(energy_eV) + sigma_excitation(energy_eV) + sigma_ionization(energy_eV)

# Pre-compute max cross-section for null-collision method
E_test = np.linspace(0.01, 100, 1000)
sigma_max = max(total_cross_section(E) for E in E_test)
# Null collision frequency
v_max = np.sqrt(2 * 100 * e_charge / m_e)  # max velocity for 100 eV electrons
nu_null = n_gas * sigma_max * v_max

# ============================================================
# Particle data structure
# ============================================================
class ParticleSpecies:
    """Container for particle positions and velocities."""
    def __init__(self, max_particles=200000):
        self.x = np.zeros(max_particles)
        self.y = np.zeros(max_particles)
        self.vx = np.zeros(max_particles)
        self.vy = np.zeros(max_particles)
        self.vz = np.zeros(max_particles)
        self.n = 0  # Active particle count
        self.max_n = max_particles

    def add(self, x, y, vx, vy, vz):
        """Add a single particle."""
        if self.n >= self.max_n:
            return
        i = self.n
        self.x[i] = x
        self.y[i] = y
        self.vx[i] = vx
        self.vy[i] = vy
        self.vz[i] = vz
        self.n += 1

    def remove(self, indices):
        """Remove particles by index (compact arrays)."""
        if len(indices) == 0:
            return
        mask = np.ones(self.n, dtype=bool)
        mask[indices] = False
        new_n = mask.sum()
        self.x[:new_n] = self.x[:self.n][mask]
        self.y[:new_n] = self.y[:self.n][mask]
        self.vx[:new_n] = self.vx[:self.n][mask]
        self.vy[:new_n] = self.vy[:self.n][mask]
        self.vz[:new_n] = self.vz[:self.n][mask]
        self.n = new_n

# ============================================================
# CIC (Cloud-In-Cell) charge weighting
# ============================================================
def deposit_charge(species, q, weight, nx, ny, dx, dy):
    """Deposit particle charge onto grid using CIC."""
    rho = np.zeros((nx + 1, ny + 1))

    for i in range(species.n):
        # Cell indices
        xi = species.x[i] / dx
        yi = species.y[i] / dy
        ix = int(xi)
        iy = int(yi)

        # Weights
        fx = xi - ix
        fy = yi - iy

        # Clamp
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)

        # Deposit
        rho[ix, iy]       += q * weight * (1-fx) * (1-fy)
        rho[ix+1, iy]     += q * weight * fx * (1-fy)
        rho[ix, iy+1]     += q * weight * (1-fx) * fy
        rho[ix+1, iy+1]   += q * weight * fx * fy

    # Normalize by cell volume (2D: area)
    rho /= (dx * dy)

    # Periodic in x
    rho[0, :] += rho[nx, :]
    rho[nx, :] = rho[0, :]

    return rho

# ============================================================
# Poisson solver (FFT in x, tridiagonal in y)
# ============================================================
def solve_poisson(rho_net, nx, ny, dx, dy, V_top, V_bottom):
    """Solve ∇²φ = -ρ/ε₀ with Dirichlet in y, periodic in x."""
    # Right-hand side
    f = -rho_net[:-1, 1:-1] / eps0  # Interior points (exclude last x=periodic, y boundaries)

    # FFT in x (periodic)
    f_hat = np.fft.fft(f, axis=0)

    # Solve tridiagonal in y for each Fourier mode
    phi_hat = np.zeros_like(f_hat)
    ny_int = ny - 1  # Interior y-points

    for kx in range(nx):
        # Eigenvalue for this Fourier mode
        lam_x = (2.0 / dx**2) * (np.cos(2 * np.pi * kx / nx) - 1.0)

        # Tridiagonal system: (d²/dy² + λx) φ_hat = f_hat
        a_diag = np.full(ny_int, 1.0 / dy**2)          # Sub-diagonal
        b_diag = np.full(ny_int, -2.0 / dy**2 + lam_x) # Main diagonal
        c_diag = np.full(ny_int, 1.0 / dy**2)          # Super-diagonal

        rhs = f_hat[kx, :].copy()
        # BC: φ(y=0) = V_bottom, φ(y=Ly) = V_top
        rhs[0] -= V_bottom / dy**2
        rhs[-1] -= V_top / dy**2

        # Thomas algorithm
        phi_hat[kx, :] = thomas_solve(a_diag, b_diag, c_diag, rhs)

    # Inverse FFT
    phi_interior = np.fft.ifft(phi_hat, axis=0).real

    # Reconstruct full potential
    phi = np.zeros((nx + 1, ny + 1))
    phi[:-1, 0] = V_bottom
    phi[:-1, -1] = V_top
    phi[:-1, 1:-1] = phi_interior
    phi[-1, :] = phi[0, :]  # Periodic

    return phi

def thomas_solve(a, b, c, d):
    """Thomas algorithm for tridiagonal system."""
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n, dtype=complex)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i] * c_[i-1]
        if abs(m) < 1e-30:
            m = 1e-30
        c_[i] = c[i] / m if i < n-1 else 0
        d_[i] = (d[i] - a[i] * d_[i-1]) / m

    x = np.zeros(n, dtype=complex)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x

# ============================================================
# Electric field from potential (central difference)
# ============================================================
def compute_E_field(phi, dx, dy, nx, ny):
    """E = -∇φ using central differences."""
    Ex = np.zeros((nx + 1, ny + 1))
    Ey = np.zeros((nx + 1, ny + 1))

    # Interior
    Ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dx)
    Ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dy)

    # Periodic x boundaries
    Ex[0, :] = -(phi[1, :] - phi[-2, :]) / (2 * dx)
    Ex[-1, :] = Ex[0, :]

    # y boundaries (one-sided)
    Ey[:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
    Ey[:, -1] = -(phi[:, -1] - phi[:, -2]) / dy

    return Ex, Ey

# ============================================================
# Field interpolation to particle position (CIC)
# ============================================================
def interpolate_field(Ex, Ey, species, dx, dy, nx, ny):
    """Interpolate E-field to particle positions using CIC."""
    ex_p = np.zeros(species.n)
    ey_p = np.zeros(species.n)

    for i in range(species.n):
        xi = species.x[i] / dx
        yi = species.y[i] / dy
        ix = int(xi)
        iy = int(yi)
        fx = xi - ix
        fy = yi - iy
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)

        ex_p[i] = (Ex[ix, iy] * (1-fx)*(1-fy) +
                   Ex[ix+1, iy] * fx*(1-fy) +
                   Ex[ix, iy+1] * (1-fx)*fy +
                   Ex[ix+1, iy+1] * fx*fy)
        ey_p[i] = (Ey[ix, iy] * (1-fx)*(1-fy) +
                   Ey[ix+1, iy] * fx*(1-fy) +
                   Ey[ix, iy+1] * (1-fx)*fy +
                   Ey[ix+1, iy+1] * fx*fy)

    return ex_p, ey_p

# ============================================================
# Boris pusher (electrostatic: no B-field)
# ============================================================
def push_particles(species, ex_p, ey_p, q, m, dt):
    """Leapfrog/Boris push (E-field only, no B-field)."""
    n = species.n
    qdt_2m = q * dt / (2.0 * m)

    # Half-step acceleration
    species.vx[:n] += qdt_2m * ex_p * 2  # Full step since no B
    species.vy[:n] += qdt_2m * ey_p * 2

    # Position update
    species.x[:n] += species.vx[:n] * dt
    species.y[:n] += species.vy[:n] * dt

# ============================================================
# Boundary conditions for particles
# ============================================================
def apply_particle_bc(species, Lx, Ly):
    """Periodic in x, absorbing in y (electrodes)."""
    n = species.n
    # Periodic x
    species.x[:n] = species.x[:n] % Lx

    # Absorbing y: collect indices to remove
    remove = []
    for i in range(n):
        if species.y[i] < 0 or species.y[i] > Ly:
            remove.append(i)

    return remove

# ============================================================
# Monte Carlo Collisions (MCC) — null-collision method
# ============================================================
def mcc_collisions(electrons, ions, dt, n_gas):
    """Perform MCC for electron-neutral collisions."""
    n_e = electrons.n
    if n_e == 0:
        return 0

    # Probability of null collision per particle per dt
    P_null = 1 - np.exp(-nu_null * dt)

    # Select particles for collision
    rng = np.random.random(n_e)
    collision_mask = rng < P_null

    n_ionized = 0

    for i in np.where(collision_mask)[0]:
        if i >= electrons.n:
            break
        # Electron energy
        v2 = electrons.vx[i]**2 + electrons.vy[i]**2 + electrons.vz[i]**2
        energy_eV = 0.5 * m_e * v2 / e_charge

        if energy_eV < 0.01:
            continue

        # Cross-sections at this energy
        s_el = sigma_elastic(energy_eV)
        s_ex = sigma_excitation(energy_eV)
        s_iz = sigma_ionization(energy_eV)
        s_tot = s_el + s_ex + s_iz

        if s_tot < 1e-30:
            continue

        # Actual collision probability
        v = np.sqrt(v2)
        P_actual = n_gas * s_tot * v / (n_gas * sigma_max * v_max)

        if np.random.random() > P_actual:
            continue  # Null collision

        # Determine collision type
        r = np.random.random() * s_tot
        if r < s_el:
            # Elastic: isotropic scattering, small energy loss
            cos_theta = 1 - 2 * np.random.random()
            sin_theta = np.sqrt(1 - cos_theta**2)
            phi = 2 * np.pi * np.random.random()
            speed = v  # Nearly same speed (m_e << m_Ar)
            electrons.vx[i] = speed * sin_theta * np.cos(phi)
            electrons.vy[i] = speed * sin_theta * np.sin(phi)
            electrons.vz[i] = speed * cos_theta

        elif r < s_el + s_ex:
            # Excitation: lose 11.55 eV
            E_loss = 11.55
            E_new = max(energy_eV - E_loss, 0.01)
            scale = np.sqrt(E_new / max(energy_eV, 0.01))
            electrons.vx[i] *= scale
            electrons.vy[i] *= scale
            electrons.vz[i] *= scale

        else:
            # Ionization: lose 15.76 eV, create new electron + ion
            E_loss = 15.76
            E_remain = max(energy_eV - E_loss, 0.01)
            # Split remaining energy between primary and secondary electron
            E_primary = E_remain * 0.5
            E_secondary = E_remain * 0.5

            scale_p = np.sqrt(E_primary / max(energy_eV, 0.01))
            electrons.vx[i] *= scale_p
            electrons.vy[i] *= scale_p
            electrons.vz[i] *= scale_p

            # New electron (isotropic)
            v_sec = np.sqrt(2 * E_secondary * e_charge / m_e)
            cos_th = 1 - 2 * np.random.random()
            sin_th = np.sqrt(1 - cos_th**2)
            phi_s = 2 * np.pi * np.random.random()
            electrons.add(electrons.x[i], electrons.y[i],
                         v_sec * sin_th * np.cos(phi_s),
                         v_sec * sin_th * np.sin(phi_s),
                         v_sec * cos_th)

            # New ion (thermal velocity, same position)
            v_ion = np.sqrt(k_B * T_gas / m_Ar)
            ions.add(electrons.x[i], electrons.y[i],
                    np.random.normal(0, v_ion),
                    np.random.normal(0, v_ion),
                    np.random.normal(0, v_ion))

            n_ionized += 1

    return n_ionized

# ============================================================
# Initialize particles
# ============================================================
def initialize_particles(n_ppc, nx, ny, dx, dy, Lx, Ly, T_e, T_i):
    """Create uniform plasma with Maxwellian velocity distribution."""
    electrons = ParticleSpecies()
    ions = ParticleSpecies()

    v_th_e = np.sqrt(e_charge * T_e / m_e)
    v_th_i = np.sqrt(e_charge * T_i / m_Ar)

    for ix in range(nx):
        for iy in range(ny):
            for _ in range(n_ppc):
                x = (ix + np.random.random()) * dx
                y = (iy + np.random.random()) * dy
                y = max(dy * 0.1, min(y, Ly - dy * 0.1))  # Keep away from walls

                # Maxwellian velocities
                electrons.add(x, y,
                             np.random.normal(0, v_th_e),
                             np.random.normal(0, v_th_e),
                             np.random.normal(0, v_th_e))

                ions.add(x, y,
                        np.random.normal(0, v_th_i),
                        np.random.normal(0, v_th_i),
                        np.random.normal(0, v_th_i))

    return electrons, ions

# ============================================================
# Diagnostics
# ============================================================
def compute_density_profile(species, nx, ny, dx, dy, weight):
    """1D density profile along y (averaged over x)."""
    rho = deposit_charge(species, 1.0, weight, nx, ny, dx, dy)
    return rho.mean(axis=0)  # Average over x

def compute_eedf(electrons, n_bins=50, E_max=50):
    """Electron Energy Distribution Function."""
    n = electrons.n
    if n == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    v2 = electrons.vx[:n]**2 + electrons.vy[:n]**2 + electrons.vz[:n]**2
    energies = 0.5 * m_e * v2 / e_charge  # [eV]
    bins = np.linspace(0, E_max, n_bins + 1)
    hist, _ = np.histogram(energies, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Normalize: f(E) ~ dN/dE / sqrt(E)
    dE = bins[1] - bins[0]
    eedf = hist / (n * dE)
    return bin_centers, eedf

def compute_iedf_at_electrode(ions, Ly, dy, n_bins=50, E_max=100):
    """Ion Energy Distribution at grounded electrode (y=0)."""
    # Collect ions near bottom electrode
    n = ions.n
    mask = ions.y[:n] < 2 * dy
    if mask.sum() == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    v2 = ions.vx[:n][mask]**2 + ions.vy[:n][mask]**2 + ions.vz[:n][mask]**2
    energies = 0.5 * m_Ar * v2 / e_charge
    bins = np.linspace(0, E_max, n_bins + 1)
    hist, _ = np.histogram(energies, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dE = bins[1] - bins[0]
    if mask.sum() > 0:
        iedf = hist / (mask.sum() * dE)
    else:
        iedf = hist * 0
    return bin_centers, iedf

# ============================================================
# Visualization
# ============================================================
def plot_results(phi, electrons, ions, nx, ny, dx, dy, Lx, Ly, weight, save_path):
    """6-panel figure: density, potential, EEDF, IEDF, phase space, E-field."""
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle('2D PIC-MCC Plasma Simulation — Ar CCP Discharge (TEL)',
                 fontsize=15, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Electron and ion density profiles
    ax1 = fig.add_subplot(gs[0, 0])
    ne_prof = compute_density_profile(electrons, nx, ny, dx, dy, weight)
    ni_prof = compute_density_profile(ions, nx, ny, dx, dy, weight)
    y_grid = np.linspace(0, Ly*100, ny+1)  # [cm]
    ax1.plot(y_grid, ne_prof / 1e15, 'b-', linewidth=2, label='Electron')
    ax1.plot(y_grid, ni_prof / 1e15, 'r--', linewidth=2, label='Ion (Ar⁺)')
    ax1.set_xlabel('y [cm]')
    ax1.set_ylabel('Density [×10¹⁵ m⁻³]')
    ax1.set_title('(a) Plasma Density Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Potential distribution (2D)
    ax2 = fig.add_subplot(gs[0, 1])
    x_grid = np.linspace(0, Lx*100, nx+1)
    y_grid2 = np.linspace(0, Ly*100, ny+1)
    X, Y = np.meshgrid(x_grid, y_grid2, indexing='ij')
    pcm = ax2.pcolormesh(X, Y, phi, cmap='RdBu_r', shading='auto')
    plt.colorbar(pcm, ax=ax2, label='Potential [V]')
    ax2.set_xlabel('x [cm]')
    ax2.set_ylabel('y [cm]')
    ax2.set_title('(b) Electrostatic Potential')
    ax2.set_aspect('equal')

    # (c) EEDF
    ax3 = fig.add_subplot(gs[0, 2])
    E_bins, eedf = compute_eedf(electrons, n_bins=60, E_max=50)
    ax3.semilogy(E_bins, eedf + 1e-10, 'b-', linewidth=2)
    # Maxwellian reference
    T_eff = T_e
    maxwell = (2.0 / np.sqrt(np.pi)) * (1.0/(T_eff))**1.5 * np.sqrt(E_bins) * np.exp(-E_bins/T_eff)
    if eedf.max() > 0:
        maxwell *= eedf.max() / max(maxwell.max(), 1e-30)
    ax3.semilogy(E_bins, maxwell + 1e-10, 'r--', linewidth=1.5, label=f'Maxwellian ({T_eff} eV)')
    ax3.set_xlabel('Energy [eV]')
    ax3.set_ylabel('EEDF [eV⁻¹]')
    ax3.set_title('(c) Electron Energy Distribution')
    ax3.set_xlim(0, 40)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (d) Phase space (y vs vy for electrons)
    ax4 = fig.add_subplot(gs[1, 0])
    n_plot = min(5000, electrons.n)
    idx = np.random.choice(electrons.n, n_plot, replace=False) if electrons.n > n_plot else np.arange(electrons.n)
    vy_eV = 0.5 * m_e * electrons.vy[idx]**2 / e_charge
    vy_signed = np.sign(electrons.vy[idx]) * np.sqrt(vy_eV)
    ax4.scatter(electrons.y[idx]*100, vy_signed, s=0.5, alpha=0.3, c='blue')
    ax4.set_xlabel('y [cm]')
    ax4.set_ylabel('v_y [√eV]')
    ax4.set_title('(d) Electron Phase Space (y - v_y)')
    ax4.grid(True, alpha=0.3)

    # (e) Ion Energy Distribution at electrode
    ax5 = fig.add_subplot(gs[1, 1])
    E_ion, iedf = compute_iedf_at_electrode(ions, Ly, dy, n_bins=50, E_max=80)
    ax5.plot(E_ion, iedf, 'r-', linewidth=2)
    ax5.fill_between(E_ion, iedf, alpha=0.3, color='red')
    ax5.set_xlabel('Energy [eV]')
    ax5.set_ylabel('IEDF [eV⁻¹]')
    ax5.set_title('(e) Ion Energy Distribution at Electrode')
    ax5.grid(True, alpha=0.3)

    # (f) Electric field profile (y-component, averaged over x)
    ax6 = fig.add_subplot(gs[1, 2])
    Ex, Ey = compute_E_field(phi, dx, dy, nx, ny)
    Ey_avg = Ey.mean(axis=0)
    y_g = np.linspace(0, Ly*100, ny+1)
    ax6.plot(y_g, Ey_avg / 1e3, 'g-', linewidth=2)
    ax6.set_xlabel('y [cm]')
    ax6.set_ylabel('E_y [kV/m]')
    ax6.set_title('(f) Electric Field Profile (y-direction)')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='k', linewidth=0.5)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# Main simulation loop
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("2D PIC-MCC Plasma Simulation — Ar CCP Discharge")
    print("Portfolio: Tokyo Electron (TEL)")
    print("=" * 60)

    print(f"\nDomain: {Lx*100:.1f} x {Ly*100:.1f} cm, Grid: {nx}x{ny}")
    print(f"Plasma: n₀={n0:.1e} m⁻³, Te={T_e} eV, p={p_gas} Pa")
    print(f"RF: V={V_rf} V, f={f_rf/1e6:.2f} MHz")
    print(f"Particles/cell: {n_ppc}, weight: {weight:.2e}")
    print(f"dt = {dt:.3e} s, {n_steps} steps")

    # Initialize
    print("\nInitializing particles...")
    electrons, ions = initialize_particles(n_ppc, nx, ny, dx, dy, Lx, Ly, T_e, T_i)
    print(f"  Electrons: {electrons.n}, Ions: {ions.n}")

    # Main loop
    print("\nRunning simulation...")
    total_ionizations = 0

    for step in range(n_steps):
        t = step * dt

        # RF voltage on top electrode
        V_top = V_rf * np.sin(2 * np.pi * f_rf * t)
        V_bottom = 0.0

        # 1. Charge deposition
        rho_e = deposit_charge(electrons, -e_charge, weight, nx, ny, dx, dy)
        rho_i = deposit_charge(ions, e_charge, weight, nx, ny, dx, dy)
        rho_net = rho_e + rho_i

        # 2. Solve Poisson
        phi = solve_poisson(rho_net, nx, ny, dx, dy, V_top, V_bottom)

        # 3. Electric field
        Ex, Ey = compute_E_field(phi, dx, dy, nx, ny)

        # 4. Interpolate field to particles
        ex_e, ey_e = interpolate_field(Ex, Ey, electrons, dx, dy, nx, ny)
        ex_i, ey_i = interpolate_field(Ex, Ey, ions, dx, dy, nx, ny)

        # 5. Push particles (Boris)
        push_particles(electrons, ex_e, ey_e, -e_charge, m_e, dt)
        push_particles(ions, ex_i, ey_i, e_charge, m_Ar, dt)

        # 6. Boundary conditions
        e_remove = apply_particle_bc(electrons, Lx, Ly)
        i_remove = apply_particle_bc(ions, Lx, Ly)

        # Secondary electron emission from ion impact
        for idx in i_remove:
            if idx < ions.n and ions.y[idx] <= 0:
                if np.random.random() < gamma_see:
                    v_th = np.sqrt(e_charge * 2.0 / m_e)  # ~2 eV secondary
                    electrons.add(ions.x[idx], dy * 0.1,
                                 np.random.normal(0, v_th),
                                 abs(np.random.normal(0, v_th)),
                                 np.random.normal(0, v_th))

        electrons.remove(e_remove)
        ions.remove(i_remove)

        # 7. Collisions (MCC)
        n_iz = mcc_collisions(electrons, ions, dt, n_gas)
        total_ionizations += n_iz

        # Progress
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{n_steps}: Ne={electrons.n}, Ni={ions.n}, "
                  f"ionizations={total_ionizations}")

    print(f"\nSimulation complete.")
    print(f"  Final: Ne={electrons.n}, Ni={ions.n}")
    print(f"  Total ionizations: {total_ionizations}")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "pic_mcc_results.png")
    print("\nPlotting results...")
    plot_results(phi, electrons, ions, nx, ny, dx, dy, Lx, Ly, weight, save_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
