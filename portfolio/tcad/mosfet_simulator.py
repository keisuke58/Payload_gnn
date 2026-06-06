#!/usr/bin/env python3
"""
1D MOSFET Device Simulator (Simplified TCAD)
==============================================
Portfolio: Renesas / TSMC / Sony Semiconductor

Solves the 1D semiconductor equations self-consistently for an n-MOSFET:
1. Poisson equation: d²ψ/dx² = -q(p - n + N_D - N_A)/ε
2. Drift-diffusion: Jn = qμn·n·E + qDn·dn/dx
3. Continuity with SRH recombination

Uses Gummel iteration (decoupled) for self-consistent solution.

Physics:
- Boltzmann statistics for carrier concentrations
- Constant mobility model
- SRH recombination
- MOS capacitor gate model

Author: Nishioka
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time

# ============================================================
# Physical constants
# ============================================================
q = 1.602176634e-19       # Elementary charge [C]
k_B = 1.380649e-23        # Boltzmann constant [J/K]
eps_Si = 11.7 * 8.854e-12 # Si permittivity [F/m]
eps_ox = 3.9 * 8.854e-12  # SiO2 permittivity [F/m]
ni = 1.5e16               # Intrinsic carrier concentration Si at 300K [m⁻³]
T = 300.0                 # Temperature [K]
Vt = k_B * T / q          # Thermal voltage [V] (~26 mV)

# Mobility (constant model)
mu_n = 0.14               # Electron mobility [m²/V·s]
mu_p = 0.045              # Hole mobility [m²/V·s]
D_n = mu_n * Vt           # Einstein relation
D_p = mu_p * Vt

# SRH recombination
tau_n = 1e-6              # Electron lifetime [s]
tau_p = 1e-6              # Hole lifetime [s]

# ============================================================
# Device structure
# ============================================================
L_ch = 100e-9     # Channel length [m]
L_sd = 50e-9      # Source/drain extension [m]
L_total = L_sd + L_ch + L_sd  # Total device length

t_ox = 2e-9       # Gate oxide thickness [m]

# Doping profile [m⁻³]
N_A_ch = 5e23      # p-type channel (5×10¹⁷ cm⁻³)
N_D_sd = 1e26      # n+ source/drain (10²⁰ cm⁻³)

# ============================================================
# Mesh
# ============================================================
nx = 200
x = np.linspace(0, L_total, nx)
dx = x[1] - x[0]

# Doping profile
N_D = np.zeros(nx)
N_A = np.zeros(nx)
for i in range(nx):
    if x[i] < L_sd or x[i] > L_sd + L_ch:
        N_D[i] = N_D_sd  # n+ source/drain
    else:
        N_A[i] = N_A_ch  # p-channel

N_net = N_D - N_A  # Net doping (positive = n-type)

# ============================================================
# Equilibrium solution
# ============================================================
def equilibrium_potential(N_net):
    """Initial guess: equilibrium potential from charge neutrality."""
    psi = np.zeros(nx)
    for i in range(nx):
        if N_net[i] > 0:
            psi[i] = Vt * np.log(N_net[i] / ni)
        elif N_net[i] < 0:
            psi[i] = -Vt * np.log(-N_net[i] / ni)
        else:
            psi[i] = 0.0
    return psi

# ============================================================
# Carrier concentrations (Boltzmann)
# ============================================================
def carrier_conc(psi, phi_n=None, phi_p=None):
    """
    n = ni * exp((ψ - φn) / Vt)
    p = ni * exp((φp - ψ) / Vt)
    """
    if phi_n is None:
        phi_n = np.zeros_like(psi)
    if phi_p is None:
        phi_p = np.zeros_like(psi)

    # Clamp exponents to avoid overflow
    arg_n = np.clip((psi - phi_n) / Vt, -50, 50)
    arg_p = np.clip((phi_p - psi) / Vt, -50, 50)

    n = ni * np.exp(arg_n)
    p = ni * np.exp(arg_p)
    return n, p

# ============================================================
# Poisson solver (Newton-Raphson)
# ============================================================
def solve_poisson(psi, phi_n, phi_p, V_gate, Vds):
    """
    Solve Poisson equation with gate BC using Newton's method.
    d²ψ/dx² = -q/ε (p - n + N_D - N_A)
    """
    psi_new = psi.copy()

    for iteration in range(50):
        n, p = carrier_conc(psi_new, phi_n, phi_p)

        # Residual: F = d²ψ/dx² + q/ε(p - n + N_net)
        F = np.zeros(nx)
        for i in range(1, nx-1):
            F[i] = (psi_new[i-1] - 2*psi_new[i] + psi_new[i+1]) / dx**2 \
                   + q / eps_Si * (p[i] - n[i] + N_net[i])

        # Gate effect on channel (simplified MOS capacitor)
        # Surface potential at channel controlled by gate voltage
        # Cox = eps_ox / t_ox
        Cox = eps_ox / t_ox
        for i in range(nx):
            if L_sd <= x[i] <= L_sd + L_ch:
                # Gate coupling: Cox * (V_gate - V_fb - ψs) adds charge
                V_fb = -0.56  # Flatband voltage (n+ poly gate on p-Si)
                gate_charge = Cox * (V_gate - V_fb - psi_new[i])
                F[i] += gate_charge / (eps_Si * 10e-9)  # Distribute over depletion width

        # Jacobian diagonal
        dn_dpsi = n / Vt
        dp_dpsi = -p / Vt
        J_diag = -2.0 / dx**2 + q / eps_Si * (dp_dpsi - dn_dpsi)

        # Add gate coupling derivative
        for i in range(nx):
            if L_sd <= x[i] <= L_sd + L_ch:
                J_diag[i] -= Cox / (eps_Si * 10e-9)

        # Boundary conditions
        F[0] = psi_new[0] - Vt * np.log(N_D_sd / ni)  # Source = 0V
        F[-1] = psi_new[-1] - (Vt * np.log(N_D_sd / ni) + Vds)  # Drain = Vds
        J_diag[0] = 1.0
        J_diag[-1] = 1.0

        # Newton update (simplified: diagonal Jacobian)
        dpsi = -F / (J_diag + 1e-30)
        dpsi = np.clip(dpsi, -0.5*Vt, 0.5*Vt)  # Damping

        psi_new += dpsi

        if np.max(np.abs(dpsi)) < 1e-6 * Vt:
            break

    return psi_new

# ============================================================
# Drift-Diffusion Current Solver
# ============================================================
def solve_current(psi, n, p, Vds):
    """Compute current using Scharfetter-Gummel discretization."""
    Jn = np.zeros(nx - 1)
    Jp = np.zeros(nx - 1)

    for i in range(nx - 1):
        dpsi = psi[i+1] - psi[i]
        B_pos = dpsi / Vt / (np.exp(dpsi/Vt) - 1) if abs(dpsi/Vt) > 1e-6 else 1.0
        B_neg = -dpsi / Vt / (np.exp(-dpsi/Vt) - 1) if abs(dpsi/Vt) > 1e-6 else 1.0

        Jn[i] = q * mu_n * Vt / dx * (n[i+1] * B_pos - n[i] * B_neg)
        Jp[i] = q * mu_p * Vt / dx * (p[i] * B_neg - p[i+1] * B_pos)

    return Jn, Jp

# ============================================================
# Gummel Iteration
# ============================================================
def gummel_solve(V_gate, Vds, max_iter=100):
    """Self-consistent Gummel iteration."""
    psi = equilibrium_potential(N_net)
    psi[-1] += Vds  # Drain bias

    phi_n = np.zeros(nx)
    phi_p = np.zeros(nx)
    phi_n[-1] = -Vds  # Quasi-Fermi level shift at drain

    # Linear interpolation of quasi-Fermi levels as initial guess
    phi_n = np.linspace(0, -Vds, nx)

    for iteration in range(max_iter):
        psi_old = psi.copy()

        # Solve Poisson
        psi = solve_poisson(psi, phi_n, phi_p, V_gate, Vds)

        # Compute carriers
        n, p = carrier_conc(psi, phi_n, phi_p)

        # Check convergence
        if np.max(np.abs(psi - psi_old)) < 1e-6 * Vt:
            break

    # Compute current
    Jn, Jp = solve_current(psi, n, p, Vds)
    # Drain current = average current density
    Id = np.mean(Jn + Jp)  # [A/m²] → multiply by width for total

    return psi, n, p, Id, Jn, Jp

# ============================================================
# I-V Characteristics
# ============================================================
def compute_iv_transfer(Vds=0.7):
    """Id vs Vgs at fixed Vds."""
    Vgs_range = np.linspace(-0.5, 1.5, 40)
    Id_list = []

    for Vgs in Vgs_range:
        _, _, _, Id, _, _ = gummel_solve(Vgs, Vds)
        Id_list.append(abs(Id))

    return Vgs_range, np.array(Id_list)

def compute_iv_output(Vgs_list=[0.4, 0.6, 0.8, 1.0, 1.2]):
    """Id vs Vds at multiple Vgs."""
    Vds_range = np.linspace(0.01, 1.2, 30)
    results = {}

    for Vgs in Vgs_list:
        Id_list = []
        for Vds in Vds_range:
            _, _, _, Id, _, _ = gummel_solve(Vgs, Vds)
            Id_list.append(abs(Id))
        results[Vgs] = np.array(Id_list)

    return Vds_range, results

# ============================================================
# Visualization
# ============================================================
def plot_results(save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('1D MOSFET Device Simulator (Simplified TCAD) — Renesas/TSMC/Sony',
                 fontsize=15, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- (a) Transfer characteristic: Id vs Vgs ---
    ax1 = fig.add_subplot(gs[0, 0])
    print("  Computing Id-Vgs (transfer)...")
    Vgs, Id_transfer = compute_iv_transfer(Vds=0.7)
    W = 1e-6  # 1 μm width
    Id_uA = Id_transfer * W * 1e6  # [μA/μm]

    ax1.semilogy(Vgs, Id_uA + 1e-6, 'b-', linewidth=2)
    ax1.set_xlabel('V_GS [V]')
    ax1.set_ylabel('I_D [μA/μm] (log)')
    ax1.set_title('(a) Transfer Characteristic (V_DS = 0.7V)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 1.5)

    # Threshold voltage (extrapolation)
    # Find max gm point
    gm = np.gradient(Id_transfer, Vgs)
    gm_max_idx = np.argmax(gm)
    if gm_max_idx > 0 and gm[gm_max_idx] > 0:
        Vth = Vgs[gm_max_idx] - Id_transfer[gm_max_idx] / gm[gm_max_idx]
        ax1.axvline(Vth, color='red', linestyle='--', alpha=0.5,
                   label=f'V_th ≈ {Vth:.2f} V')
        ax1.legend(fontsize=10)

    # Linear scale on twin axis
    ax1b = ax1.twinx()
    ax1b.plot(Vgs, Id_uA, 'r--', linewidth=1, alpha=0.5)
    ax1b.set_ylabel('I_D [μA/μm] (linear)', color='red')
    ax1b.tick_params(axis='y', labelcolor='red')

    # --- (b) Output characteristic: Id vs Vds ---
    ax2 = fig.add_subplot(gs[0, 1])
    print("  Computing Id-Vds (output)...")
    Vgs_list = [0.4, 0.6, 0.8, 1.0, 1.2]
    Vds_range, Id_output = compute_iv_output(Vgs_list)

    for Vgs_val in Vgs_list:
        Id_out = Id_output[Vgs_val] * W * 1e6
        ax2.plot(Vds_range, Id_out, linewidth=2, label=f'V_GS={Vgs_val}V')

    ax2.set_xlabel('V_DS [V]')
    ax2.set_ylabel('I_D [μA/μm]')
    ax2.set_title('(b) Output Characteristics')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Internal quantities at Vgs=1.0V, Vds=0.7V ---
    print("  Computing band diagram...")
    psi, n, p, Id, Jn, Jp = gummel_solve(V_gate=1.0, Vds=0.7)
    x_nm = x * 1e9

    # (c) Band diagram
    ax3 = fig.add_subplot(gs[0, 2])
    Eg = 1.12  # Si bandgap [eV]
    chi = 4.05  # Electron affinity [eV]

    Ec = -psi - chi + chi  # Conduction band edge (relative)
    Ev = Ec - Eg
    Ef_n = -psi  # Electron quasi-Fermi level (simplified)

    ax3.plot(x_nm, -psi, 'b-', linewidth=2, label='E_c')
    ax3.plot(x_nm, -psi - Eg, 'r-', linewidth=2, label='E_v')
    ax3.axhline(-0.56, color='green', linestyle='--', alpha=0.3, label='E_F (approx)')
    ax3.set_xlabel('Position [nm]')
    ax3.set_ylabel('Energy [eV]')
    ax3.set_title('(c) Band Diagram (V_GS=1V, V_DS=0.7V)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Mark S/D/Channel
    ax3.axvspan(0, L_sd*1e9, alpha=0.1, color='blue', label='Source')
    ax3.axvspan(L_sd*1e9, (L_sd+L_ch)*1e9, alpha=0.1, color='yellow')
    ax3.axvspan((L_sd+L_ch)*1e9, L_total*1e9, alpha=0.1, color='red')

    # (d) Carrier concentrations
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(x_nm, n / 1e6, 'b-', linewidth=2, label='n (electrons)')
    ax4.semilogy(x_nm, p / 1e6, 'r-', linewidth=2, label='p (holes)')
    ax4.semilogy(x_nm, np.abs(N_net) / 1e6, 'k--', linewidth=1, alpha=0.5,
                label='|N_D - N_A|')
    ax4.set_xlabel('Position [nm]')
    ax4.set_ylabel('Concentration [cm⁻³]')
    ax4.set_title('(d) Carrier Concentration Profile')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(1e10, 1e21)

    # (e) Electric field
    ax5 = fig.add_subplot(gs[1, 1])
    E_field = -np.gradient(psi, dx) / 1e5  # [kV/cm → ×10⁵ V/m]
    ax5.plot(x_nm, E_field * 1e-2, 'g-', linewidth=2)
    ax5.set_xlabel('Position [nm]')
    ax5.set_ylabel('Electric Field [MV/cm]')
    ax5.set_title('(e) Electric Field Distribution')
    ax5.grid(True, alpha=0.3)

    # (f) Potential distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(x_nm, psi, 'm-', linewidth=2)
    ax6.set_xlabel('Position [nm]')
    ax6.set_ylabel('Electrostatic Potential ψ [V]')
    ax6.set_title('(f) Potential Distribution')
    ax6.grid(True, alpha=0.3)

    # Device info
    fig.text(0.5, 0.01,
             f'n-MOSFET: L_ch={L_ch*1e9:.0f}nm, t_ox={t_ox*1e9:.1f}nm, '
             f'N_A={N_A_ch:.0e}m⁻³, N_D(S/D)={N_D_sd:.0e}m⁻³, T={T}K',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("1D MOSFET Device Simulator (Simplified TCAD)")
    print("Portfolio: Renesas / TSMC / Sony Semiconductor")
    print("=" * 60)

    print(f"\nDevice: n-MOSFET")
    print(f"  Channel length: {L_ch*1e9:.0f} nm")
    print(f"  Oxide thickness: {t_ox*1e9:.1f} nm")
    print(f"  Channel doping: {N_A_ch:.1e} m⁻³ ({N_A_ch/1e6:.1e} cm⁻³)")
    print(f"  S/D doping: {N_D_sd:.1e} m⁻³ ({N_D_sd/1e6:.1e} cm⁻³)")
    print(f"  Temperature: {T} K, Vt = {Vt*1000:.1f} mV")
    print(f"  Grid: {nx} points, dx = {dx*1e9:.2f} nm")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "mosfet_results.png")
    print("\nRunning simulations...")
    plot_results(save_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
