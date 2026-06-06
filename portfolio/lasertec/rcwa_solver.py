#!/usr/bin/env python3
"""
RCWA (Rigorous Coupled-Wave Analysis) Solver
==============================================
Portfolio: Lasertec Corporation — EUV Mask Inspection

Solves Maxwell's equations for electromagnetic diffraction from
periodic structures using the Fourier modal method.

Theory:
- Expand permittivity and fields in Fourier series
- Solve eigenvalue problem in each layer
- Cascade layers using S-matrix (Redheffer star product)
- Compute diffraction efficiencies for all orders

Application:
- EUV mask: Mo/Si multilayer Bragg reflector + TaBN absorber grating
- Compute reflectivity, diffraction orders, wavelength/angle dependence

References:
- Moharam & Gaylord, JOSA A (1995)
- Li, JOSA A 13, 1024 (1996) — Fourier factorization rules

Author: Nishioka
"""

import numpy as np
from scipy.linalg import eig, inv, block_diag
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import os
import time

# ============================================================
# Physical constants
# ============================================================
c0 = 2.99792458e8
eps0 = 8.854187817e-12
mu0 = 4e-7 * np.pi

# ============================================================
# Material database at 13.5 nm (EUV)
# Complex refractive index n = 1 - δ + iβ
# Data from CXRO (Center for X-Ray Optics, LBL)
# ============================================================
MATERIALS_EUV = {
    'vacuum':  1.0 + 0j,
    'Mo':      1.0 - 0.0764 + 0.0063j,   # Molybdenum at 13.5 nm
    'Si':      1.0 - 0.0001 + 0.0018j,   # Silicon at 13.5 nm
    'Ru':      1.0 - 0.0460 + 0.0070j,   # Ruthenium capping
    'TaBN':    1.0 - 0.0395 + 0.0098j,   # Tantalum Boron Nitride absorber
    'SiO2':    1.0 - 0.0120 + 0.0015j,   # Silicon dioxide substrate
}

# ============================================================
# RCWA Solver Class
# ============================================================
class RCWASolver:
    """
    1D RCWA solver for TE and TM polarization.
    Grating periodicity in x-direction, layers stacked in z.
    """

    def __init__(self, n_orders=11):
        """
        n_orders: total number of Fourier orders (should be odd).
        Order indices: -N, ..., -1, 0, 1, ..., N  where N = (n_orders-1)/2
        """
        self.n_orders = n_orders
        self.N = (n_orders - 1) // 2
        self.orders = np.arange(-self.N, self.N + 1)

    def _toeplitz_permittivity(self, eps_profile, n_harmonics):
        """
        Build Toeplitz matrix of Fourier coefficients of permittivity.
        eps_profile: 1D array of permittivity across one period.
        """
        n_pts = len(eps_profile)
        # Compute Fourier coefficients via FFT
        eps_fft = np.fft.fft(eps_profile) / n_pts

        n = self.n_orders
        # Build Toeplitz matrix: E[i,j] = eps_hat[i-j]
        E = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                idx = (self.orders[i] - self.orders[j]) % n_pts
                E[i, j] = eps_fft[idx]

        return E

    def _solve_layer_eigenmodes(self, E_toep, kx_norm, polarization='TE'):
        """
        Solve eigenvalue problem in a grating layer.

        For TE (s-pol): eigenvalue matrix A = kx² - E
        For TM (p-pol): eigenvalue matrix A = E⁻¹(kx² E⁻¹ - I)  (Li's rules)

        Returns eigenvalues (kz²) and eigenvectors.
        """
        n = self.n_orders
        Kx = np.diag(kx_norm)  # Diagonal matrix of kx/k0

        if polarization == 'TE':
            # A = Kx² - E
            A = Kx @ Kx - E_toep
        else:
            # TM: need inverse of E_toep
            E_inv = inv(E_toep)
            A = Kx @ E_inv @ Kx - np.eye(n)

        # Eigenvalue decomposition
        eigenvalues, W = eig(A)

        # kz = sqrt(-eigenvalue) * k0
        # Choose branch with Im(kz) > 0 or Re(kz) > 0 for propagation
        kz_sq = -eigenvalues
        kz_norm = np.sqrt(kz_sq.astype(complex))
        # Ensure forward-propagating or decaying
        for i in range(len(kz_norm)):
            if kz_norm[i].imag < -1e-10:
                kz_norm[i] = -kz_norm[i]
            elif abs(kz_norm[i].imag) < 1e-10 and kz_norm[i].real < 0:
                kz_norm[i] = -kz_norm[i]

        # V matrix (for field matching)
        Q = np.diag(kz_norm)
        if polarization == 'TE':
            V = W @ Q
        else:
            V = E_inv @ W @ Q

        return kz_norm, W, V

    def _solve_uniform_layer(self, eps_r, kx_norm, polarization='TE'):
        """Eigenmodes for a uniform (non-grating) layer."""
        n = self.n_orders
        E_toep = eps_r * np.eye(n, dtype=complex)
        return self._solve_layer_eigenmodes(E_toep, kx_norm, polarization)

    def _smatrix_layer(self, kz_norm, W, V, thickness_norm, W_ref, V_ref):
        """
        Compute S-matrix for a single layer.
        thickness_norm: layer thickness * k0
        """
        n = self.n_orders

        # Phase matrix
        phase = np.exp(1j * kz_norm * thickness_norm)
        X = np.diag(phase)

        # Interface matrices
        A = inv(W) @ W_ref + inv(V) @ V_ref
        B = inv(W) @ W_ref - inv(V) @ V_ref

        A_inv = inv(A)
        D = A - X @ B @ A_inv @ X @ B

        S11 = inv(D) @ (X @ B @ A_inv @ X @ A - B)
        S12 = inv(D) @ X @ (A - B @ A_inv @ B)
        S21 = S12.copy()
        S22 = S11.copy()

        return S11, S12, S21, S22

    def _redheffer_star(self, Sa, Sb):
        """
        Redheffer star product for cascading S-matrices.
        Sa = (S11a, S12a, S21a, S22a)
        Sb = (S11b, S12b, S21b, S22b)
        """
        S11a, S12a, S21a, S22a = Sa
        S11b, S12b, S21b, S22b = Sb
        n = S11a.shape[0]
        I = np.eye(n, dtype=complex)

        D = inv(I - S11b @ S22a)
        F = inv(I - S22a @ S11b)

        S11 = S11a + S12a @ D @ S11b @ S21a
        S12 = S12a @ D @ S12b
        S21 = S21b @ F @ S21a
        S22 = S22b + S21b @ F @ S22a @ S12b

        return S11, S12, S21, S22

    def solve(self, layers, period, wavelength, theta_deg=0.0,
              polarization='TE', n_inc=1.0, n_sub=1.0):
        """
        Solve RCWA for a stack of layers.

        layers: list of dicts:
            {'type': 'uniform', 'eps_r': complex, 'thickness': float}  or
            {'type': 'grating', 'eps_profile': array, 'thickness': float}

        period: grating period [m]
        wavelength: incident wavelength [m]
        theta_deg: angle of incidence [degrees]
        polarization: 'TE' or 'TM'
        n_inc: refractive index of incidence medium
        n_sub: refractive index of substrate

        Returns: dict with diffraction efficiencies
        """
        k0 = 2 * np.pi / wavelength
        n = self.n_orders
        theta = np.radians(theta_deg)

        # Normalized kx for each diffraction order
        # kx_m = n_inc * sin(θ) + m * λ/Λ
        kx_norm = n_inc * np.sin(theta) + self.orders * wavelength / period

        # Reference (free-space) eigenmodes
        kz_ref = np.sqrt((n_inc**2 - kx_norm**2).astype(complex))
        for i in range(n):
            if kz_ref[i].imag < -1e-10:
                kz_ref[i] = -kz_ref[i]

        W_ref = np.eye(n, dtype=complex)
        if polarization == 'TE':
            V_ref = np.diag(kz_ref)
        else:
            V_ref = np.diag(kz_ref / n_inc**2)

        # Substrate eigenmodes
        eps_sub = n_sub**2
        kz_sub = np.sqrt((eps_sub - kx_norm**2).astype(complex))
        for i in range(n):
            if kz_sub[i].imag < -1e-10:
                kz_sub[i] = -kz_sub[i]

        W_sub = np.eye(n, dtype=complex)
        if polarization == 'TE':
            V_sub = np.diag(kz_sub)
        else:
            V_sub = np.diag(kz_sub / eps_sub)

        # Initialize global S-matrix (identity)
        I = np.eye(n, dtype=complex)
        Z = np.zeros((n, n), dtype=complex)
        S_global = (Z.copy(), I.copy(), I.copy(), Z.copy())

        # Process each layer
        for layer in layers:
            thickness_norm = layer['thickness'] * k0

            if layer['type'] == 'uniform':
                eps_r = layer['eps_r']
                E_toep = eps_r * np.eye(n, dtype=complex)
            else:
                # Grating: build Toeplitz from profile
                E_toep = self._toeplitz_permittivity(layer['eps_profile'],
                                                      2 * self.N + 1)

            kz, W, V = self._solve_layer_eigenmodes(E_toep, kx_norm, polarization)

            S_layer = self._smatrix_layer(kz, W, V, thickness_norm, W_ref, V_ref)
            S_global = self._redheffer_star(S_global, S_layer)

        # Incident wave (0th order only)
        inc = np.zeros(n, dtype=complex)
        inc[self.N] = 1.0  # 0th order

        # Reflected and transmitted amplitudes
        r = S_global[0] @ inc
        t = S_global[2] @ inc

        # Diffraction efficiencies
        R_eff = np.zeros(n)
        T_eff = np.zeros(n)

        for m in range(n):
            R_eff[m] = np.real(kz_ref[m] / kz_ref[self.N]) * np.abs(r[m])**2
            T_eff[m] = np.real(kz_sub[m] / kz_ref[self.N]) * np.abs(t[m])**2

        return {
            'orders': self.orders,
            'R_eff': R_eff,
            'T_eff': T_eff,
            'R_total': np.sum(R_eff[R_eff > 0]),
            'T_total': np.sum(T_eff[T_eff > 0]),
            'r_amplitudes': r,
            't_amplitudes': t
        }

# ============================================================
# EUV Mask Model
# ============================================================
def build_euv_mask_layers(period, duty_cycle=0.5, n_bilayers=40, n_profile_pts=256):
    """
    Build layer stack for EUV mask:
    Top → Bottom:
      1. Vacuum (incidence)
      2. TaBN absorber grating
      3. Ru capping layer (2.5 nm)
      4. Mo/Si multilayer (40 bilayers, 6.9 nm period: 2.8 nm Mo + 4.1 nm Si)
      5. SiO2 substrate

    Returns list of layer dicts.
    """
    layers = []

    # 1. TaBN absorber grating (70 nm thick)
    n_pts = n_profile_pts
    eps_vacuum = 1.0
    n_TaBN = MATERIALS_EUV['TaBN']
    eps_TaBN = n_TaBN**2

    # Grating profile: absorber lines
    x = np.linspace(0, 1, n_pts, endpoint=False)
    eps_grating = np.full(n_pts, eps_vacuum, dtype=complex)
    absorber_width = int(duty_cycle * n_pts)
    eps_grating[:absorber_width] = eps_TaBN

    layers.append({
        'type': 'grating',
        'eps_profile': eps_grating,
        'thickness': 70e-9  # 70 nm
    })

    # 2. Ru capping layer (uniform, 2.5 nm)
    n_Ru = MATERIALS_EUV['Ru']
    layers.append({
        'type': 'uniform',
        'eps_r': n_Ru**2,
        'thickness': 2.5e-9
    })

    # 3. Mo/Si multilayer (40 bilayers)
    n_Mo = MATERIALS_EUV['Mo']
    n_Si = MATERIALS_EUV['Si']
    d_Mo = 2.8e-9   # Mo thickness per bilayer
    d_Si = 4.1e-9   # Si thickness per bilayer

    for _ in range(n_bilayers):
        layers.append({
            'type': 'uniform',
            'eps_r': n_Mo**2,
            'thickness': d_Mo
        })
        layers.append({
            'type': 'uniform',
            'eps_r': n_Si**2,
            'thickness': d_Si
        })

    return layers

# ============================================================
# Main analysis and visualization
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("RCWA Solver — EUV Mask Diffraction Analysis")
    print("Portfolio: Lasertec Corporation")
    print("=" * 60)

    n_orders = 15
    solver = RCWASolver(n_orders=n_orders)
    print(f"\nFourier orders: {n_orders} ({solver.orders[0]} to {solver.orders[-1]})")

    # EUV mask parameters
    period = 200e-9     # 200 nm grating period
    wavelength0 = 13.5e-9  # EUV wavelength
    n_sub_val = MATERIALS_EUV['SiO2']

    # ---- Analysis 1: Diffraction at normal incidence ----
    print("\n[1/4] Diffraction efficiency at normal incidence...")
    layers = build_euv_mask_layers(period, duty_cycle=0.5, n_bilayers=40)
    result_normal = solver.solve(layers, period, wavelength0, theta_deg=6.0,
                                  polarization='TE', n_sub=n_sub_val)
    print(f"  R_total = {result_normal['R_total']:.4f}")
    print(f"  T_total = {result_normal['T_total']:.4f}")

    # ---- Analysis 2: Defect analysis (varying duty cycle) ----
    print("[2/4] Defect analysis (duty cycle variation)...")
    duty_cycles = [0.3, 0.4, 0.5, 0.6, 0.7]
    defect_results = {}
    for dc in duty_cycles:
        layers_dc = build_euv_mask_layers(period, duty_cycle=dc, n_bilayers=40)
        res = solver.solve(layers_dc, period, wavelength0, theta_deg=6.0,
                          polarization='TE', n_sub=n_sub_val)
        defect_results[dc] = res

    # ---- Analysis 3: Wavelength scan ----
    print("[3/4] Wavelength scan (12.5 - 14.5 nm)...")
    wavelengths = np.linspace(12.5e-9, 14.5e-9, 50)
    R_vs_wl = []
    R0_vs_wl = []
    layers_scan = build_euv_mask_layers(period, duty_cycle=0.5, n_bilayers=40)
    for wl in wavelengths:
        res = solver.solve(layers_scan, period, wl, theta_deg=6.0,
                          polarization='TE', n_sub=n_sub_val)
        R_vs_wl.append(res['R_total'])
        R0_vs_wl.append(res['R_eff'][solver.N])
    R_vs_wl = np.array(R_vs_wl)
    R0_vs_wl = np.array(R0_vs_wl)

    # ---- Analysis 4: Angular scan (TE and TM) ----
    print("[4/4] Angular scan (0° - 20°, TE and TM)...")
    angles = np.linspace(0.5, 20, 40)
    R_te = []
    R_tm = []
    for ang in angles:
        res_te = solver.solve(layers_scan, period, wavelength0, theta_deg=ang,
                             polarization='TE', n_sub=n_sub_val)
        res_tm = solver.solve(layers_scan, period, wavelength0, theta_deg=ang,
                             polarization='TM', n_sub=n_sub_val)
        R_te.append(res_te['R_total'])
        R_tm.append(res_tm['R_total'])
    R_te = np.array(R_te)
    R_tm = np.array(R_tm)

    # ============================================================
    # Visualization
    # ============================================================
    print("\nPlotting results...")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('RCWA Analysis — EUV Photomask Diffraction (Lasertec)',
                 fontsize=15, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (a) Grating structure cross-section
    ax1 = fig.add_subplot(gs[0, 0])

    # Draw structure schematically
    y_base = 0
    # Substrate
    rect_sub = mpatches.FancyBboxPatch((0, y_base), 200, 50,
                boxstyle="round,pad=0", facecolor='#8B8682', edgecolor='k')
    ax1.add_patch(rect_sub)
    ax1.text(100, y_base + 25, 'SiO₂ Substrate', ha='center', va='center', fontsize=9)
    y_base += 50

    # Mo/Si multilayer (show a few)
    for i in range(6):
        # Si
        rect_si = mpatches.Rectangle((0, y_base), 200, 8,
                    facecolor='#4169E1', edgecolor='k', linewidth=0.3)
        ax1.add_patch(rect_si)
        y_base += 8
        # Mo
        rect_mo = mpatches.Rectangle((0, y_base), 200, 5.5,
                    facecolor='#808080', edgecolor='k', linewidth=0.3)
        ax1.add_patch(rect_mo)
        y_base += 5.5

    ax1.text(220, y_base - 20, '40×\nMo/Si\nBilayers', fontsize=9, va='center')
    ax1.plot([205, 215], [y_base - 40, y_base - 40], 'k-', linewidth=0.5)
    ax1.plot([205, 215], [y_base, y_base], 'k-', linewidth=0.5)

    # Ru capping
    rect_ru = mpatches.Rectangle((0, y_base), 200, 5,
                facecolor='#C0C0C0', edgecolor='k', linewidth=0.5)
    ax1.add_patch(rect_ru)
    ax1.text(100, y_base + 2.5, 'Ru Cap', ha='center', va='center', fontsize=8)
    y_base += 5

    # TaBN absorber grating
    absorber_h = 15
    dc = 0.5
    absorber_w = 200 * dc
    # Left line
    rect_abs1 = mpatches.Rectangle((0, y_base), absorber_w * 0.5, absorber_h,
                facecolor='#2F4F4F', edgecolor='k')
    ax1.add_patch(rect_abs1)
    # Right line
    rect_abs2 = mpatches.Rectangle((200 - absorber_w * 0.5, y_base), absorber_w * 0.5, absorber_h,
                facecolor='#2F4F4F', edgecolor='k')
    ax1.add_patch(rect_abs2)
    ax1.text(100, y_base + absorber_h/2, 'TaBN\nAbsorber', ha='center',
            va='center', fontsize=9, color='white')

    # Annotations
    ax1.annotate('', xy=(0, y_base + absorber_h + 5), xytext=(200, y_base + absorber_h + 5),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(100, y_base + absorber_h + 8, f'Period = {period*1e9:.0f} nm',
            ha='center', fontsize=9, color='red')

    # Incident light arrow
    ax1.annotate('EUV\n13.5 nm', xy=(100, y_base + absorber_h + 2),
                xytext=(100, y_base + absorber_h + 25),
                fontsize=10, ha='center', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax1.set_xlim(-20, 270)
    ax1.set_ylim(-10, y_base + absorber_h + 35)
    ax1.set_xlabel('x [nm (schematic)]')
    ax1.set_title('(a) EUV Mask Structure Cross-Section')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # (b) Diffraction efficiency with defect comparison
    ax2 = fig.add_subplot(gs[0, 1])
    orders = result_normal['orders']
    width = 0.15
    x_pos = np.arange(len(orders))

    for i, dc in enumerate([0.3, 0.5, 0.7]):
        res = defect_results[dc]
        offset = (i - 1) * width
        bars = ax2.bar(x_pos + offset, res['R_eff'], width,
                      label=f'DC = {dc}', alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(o) for o in orders], fontsize=8)
    ax2.set_xlabel('Diffraction Order m')
    ax2.set_ylabel('Reflective Diffraction Efficiency')
    ax2.set_title('(b) Diffraction Efficiency vs Order\n(Defect: Duty Cycle Variation)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # (c) Reflectivity vs wavelength
    ax3 = fig.add_subplot(gs[1, 0])
    wl_nm = wavelengths * 1e9
    ax3.plot(wl_nm, R_vs_wl * 100, 'b-', linewidth=2, label='Total R')
    ax3.plot(wl_nm, R0_vs_wl * 100, 'r--', linewidth=2, label='0th order R')
    ax3.axvline(13.5, color='green', linestyle=':', alpha=0.5, label='λ = 13.5 nm')
    ax3.set_xlabel('Wavelength [nm]')
    ax3.set_ylabel('Reflectivity [%]')
    ax3.set_title('(c) Reflectivity vs Wavelength\n(Mo/Si multilayer + TaBN grating)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(12.5, 14.5)

    # Add bandwidth annotation
    R_peak = R_vs_wl.max() * 100
    half_max = R_peak / 2
    above_half = wl_nm[R_vs_wl * 100 > half_max]
    if len(above_half) > 1:
        bw = above_half[-1] - above_half[0]
        ax3.annotate(f'Bandwidth ≈ {bw:.2f} nm',
                    xy=(13.5, R_peak * 0.6), fontsize=10, color='blue')

    # (d) Reflectivity vs angle (TE and TM)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(angles, R_te * 100, 'b-', linewidth=2, label='TE (s-pol)')
    ax4.plot(angles, R_tm * 100, 'r--', linewidth=2, label='TM (p-pol)')
    ax4.axvline(6.0, color='green', linestyle=':', alpha=0.5,
               label='θ = 6° (standard EUV)')
    ax4.set_xlabel('Angle of Incidence [degrees]')
    ax4.set_ylabel('Total Reflectivity [%]')
    ax4.set_title('(d) Reflectivity vs Angle of Incidence\n(TE and TM polarization)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "rcwa_results.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
