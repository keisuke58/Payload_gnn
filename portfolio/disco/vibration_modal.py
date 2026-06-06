#!/usr/bin/env python3
"""
FEM Modal Analysis of Ultrasonic Dicing Blade
==============================================
Portfolio: DISCO Corporation — Precision Cutting Equipment

Solves the generalized eigenvalue problem [K]{φ} = ω²[M]{φ}
for a thin rectangular blade (2D plane-stress Q4 elements).

Physics:
- 2D plane-stress elasticity for thin blade
- Consistent mass matrix (not lumped) for accurate high-freq modes
- Rayleigh damping for forced vibration FRF
- Boundary: clamped at left edge (hub mounting)

Author: Nishioka
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================================
# Material and geometry
# ============================================================
E = 200e9        # Young's modulus [Pa] — steel blade
nu = 0.3         # Poisson's ratio
rho = 7850.0     # Density [kg/m³]
thickness = 0.3e-3  # Blade thickness [m] (0.3 mm)

blade_length = 60e-3   # 60 mm
blade_height = 20e-3   # 20 mm

# Mesh parameters
nx, ny = 48, 16   # elements in x, y

# ============================================================
# Plane-stress constitutive matrix
# ============================================================
def constitutive_matrix(E, nu):
    """Plane-stress D matrix."""
    coeff = E / (1.0 - nu**2)
    D = coeff * np.array([
        [1.0,  nu,  0.0],
        [nu,   1.0, 0.0],
        [0.0,  0.0, (1.0 - nu) / 2.0]
    ])
    return D

# ============================================================
# Q4 element shape functions and derivatives
# ============================================================
def shape_functions(xi, eta):
    """4-node quad shape functions in natural coords (xi, eta) ∈ [-1,1]²."""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    # Derivatives dN/dxi, dN/deta  (2x4)
    dN = 0.25 * np.array([
        [-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)],
        [-(1 - xi),  -(1 + xi),   (1 + xi),   (1 - xi)]
    ])
    return N, dN

# Gauss quadrature 2x2
gauss_pts = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
gauss_wts = np.array([1.0, 1.0])

# ============================================================
# Mesh generation
# ============================================================
def generate_mesh(Lx, Ly, nx, ny):
    """Structured Q4 mesh for rectangular domain."""
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_nodes = coords.shape[0]

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elements.append([n0, n1, n2, n3])
    elements = np.array(elements)
    return coords, elements, n_nodes

# ============================================================
# Element stiffness and mass matrices
# ============================================================
def element_matrices(coords_e, D, rho, thickness):
    """Compute 8x8 element stiffness and consistent mass matrices."""
    Ke = np.zeros((8, 8))
    Me = np.zeros((8, 8))

    for i, xi in enumerate(gauss_pts):
        for j, eta in enumerate(gauss_pts):
            w = gauss_wts[i] * gauss_wts[j]
            N, dN = shape_functions(xi, eta)

            # Jacobian: J = dN · coords_e  (2x2)
            J = dN @ coords_e   # (2x4) @ (4x2) = (2x2)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Derivatives in physical coords: dN_dx = invJ @ dN
            dN_dx = invJ @ dN  # (2x4)

            # B-matrix (3x8): strain-displacement
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k]     = dN_dx[0, k]   # dN/dx
                B[1, 2*k + 1] = dN_dx[1, k]   # dN/dy
                B[2, 2*k]     = dN_dx[1, k]   # dN/dy
                B[2, 2*k + 1] = dN_dx[0, k]   # dN/dx

            # Stiffness: K += B^T D B |J| w t
            Ke += (B.T @ D @ B) * detJ * w * thickness

            # Consistent mass: M += N^T N ρ |J| w t
            # N_mat (2x8)
            N_mat = np.zeros((2, 8))
            for k in range(4):
                N_mat[0, 2*k]     = N[k]
                N_mat[1, 2*k + 1] = N[k]
            Me += (N_mat.T @ N_mat) * rho * detJ * w * thickness

    return Ke, Me

# ============================================================
# Global assembly
# ============================================================
def assemble(coords, elements, D, rho, thickness):
    """Assemble global K and M in COO sparse format."""
    n_dof = 2 * coords.shape[0]
    rows, cols, k_vals, m_vals = [], [], [], []

    for elem in elements:
        coords_e = coords[elem]  # (4, 2)
        Ke, Me = element_matrices(coords_e, D, rho, thickness)

        # DOF mapping
        dof_map = np.zeros(8, dtype=int)
        for i, node in enumerate(elem):
            dof_map[2*i]     = 2 * node
            dof_map[2*i + 1] = 2 * node + 1

        for a in range(8):
            for b in range(8):
                rows.append(dof_map[a])
                cols.append(dof_map[b])
                k_vals.append(Ke[a, b])
                m_vals.append(Me[a, b])

    K = sparse.coo_matrix((k_vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    M = sparse.coo_matrix((m_vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K, M

# ============================================================
# Boundary conditions — fixed left edge
# ============================================================
def get_fixed_dofs(coords, ny):
    """Return DOFs to fix (left edge: x=0)."""
    tol = 1e-10
    fixed = []
    for i, (x, y) in enumerate(coords):
        if x < tol:
            fixed.extend([2*i, 2*i + 1])
    return np.array(fixed, dtype=int)

# ============================================================
# Modal analysis
# ============================================================
def solve_modes(K, M, fixed_dofs, n_modes=8):
    """Solve eigenvalue problem with fixed DOFs eliminated."""
    n_dof = K.shape[0]
    free = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]

    # Shift-invert for smallest eigenvalues
    eigenvalues, eigenvectors = eigsh(K_ff, k=n_modes, M=M_ff,
                                       sigma=0.0, which='LM')

    # Sort by frequency
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Map back to full DOF
    modes_full = np.zeros((n_dof, n_modes))
    for i in range(n_modes):
        modes_full[free, i] = eigenvectors[:, i]

    freqs_hz = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)
    return freqs_hz, modes_full

# ============================================================
# Frequency Response Function (FRF)
# ============================================================
def compute_frf(K, M, fixed_dofs, coords, freq_range, n_freq=800):
    """Harmonic response: FRF at blade tip under unit force."""
    n_dof = K.shape[0]
    free = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    K_ff = K[np.ix_(free, free)]
    M_ff = M[np.ix_(free, free)]

    # Rayleigh damping: C = α*M + β*K
    # Target: ζ = 0.5% at f1=1kHz and f2=40kHz
    f1, f2 = 1000, 40000
    w1, w2 = 2*np.pi*f1, 2*np.pi*f2
    zeta = 0.005
    alpha = 2 * zeta * w1 * w2 / (w1 + w2)
    beta = 2 * zeta / (w1 + w2)
    C_ff = alpha * M_ff + beta * K_ff

    # Force at blade tip (top-right corner, y-direction)
    tip_node = np.argmax(coords[:, 0] + coords[:, 1])
    tip_dof_global = 2 * tip_node + 1  # y-direction
    # Map to free DOF index
    tip_dof_free = np.searchsorted(free, tip_dof_global)

    # Response DOF = same as force DOF
    resp_dof_free = tip_dof_free

    freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
    frf = np.zeros(n_freq, dtype=complex)

    # Force vector (unit force at tip)
    F_free = np.zeros(len(free))
    F_free[tip_dof_free] = 1.0
    F_sparse = sparse.csr_matrix(F_free.reshape(-1, 1))

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        # Dynamic stiffness: Z = K - ω²M + jωC
        Z = K_ff - omega**2 * M_ff + 1j * omega * C_ff
        # Solve Z u = F
        u = sparse.linalg.spsolve(Z.tocsc(), F_free)
        frf[i] = u[resp_dof_free]

    return freqs, frf

# ============================================================
# Visualization
# ============================================================
def plot_results(coords, elements, freqs_hz, modes, frf_freqs, frf_data, save_path):
    """Multi-panel figure: 6 mode shapes + FRF + mesh."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('FEM Modal Analysis — Ultrasonic Dicing Blade (DISCO)',
                 fontsize=16, fontweight='bold')

    # --- Mode shapes (top 2 rows, 3 columns each) ---
    n_modes_plot = min(6, len(freqs_hz))
    scale = blade_length * 0.15  # deformation magnification

    for m in range(n_modes_plot):
        ax = fig.add_subplot(3, 3, m + 1)

        ux = modes[0::2, m]
        uy = modes[1::2, m]
        mag = np.sqrt(ux**2 + uy**2)
        if mag.max() > 0:
            ux_n = ux / mag.max() * scale
            uy_n = uy / mag.max() * scale
        else:
            ux_n, uy_n = ux, uy

        deformed_x = coords[:, 0] + ux_n
        deformed_y = coords[:, 1] + uy_n

        # Create triangulation from Q4 elements
        triangles = []
        for elem in elements:
            triangles.append([elem[0], elem[1], elem[2]])
            triangles.append([elem[0], elem[2], elem[3]])
        tri = mtri.Triangulation(deformed_x, deformed_y, triangles)

        tcf = ax.tripcolor(tri, mag, shading='gouraud', cmap='jet')
        ax.set_aspect('equal')
        ax.set_title(f'Mode {m+1}: f = {freqs_hz[m]:.0f} Hz', fontsize=11)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.colorbar(tcf, ax=ax, label='|u| [arb.]', shrink=0.8)

    # --- FRF plot (bottom left, spanning 2 columns) ---
    gs = fig.add_gridspec(3, 3)
    ax_frf = fig.add_subplot(gs[2, 0:2])
    ax_frf.semilogy(frf_freqs / 1000, np.abs(frf_data), 'b-', linewidth=0.8)
    ax_frf.set_xlabel('Frequency [kHz]', fontsize=12)
    ax_frf.set_ylabel('|H(f)| [m/N]', fontsize=12)
    ax_frf.set_title('Frequency Response Function (Blade Tip, Y-direction)', fontsize=12)
    ax_frf.grid(True, alpha=0.3)

    # Mark natural frequencies
    for i, f in enumerate(freqs_hz[:n_modes_plot]):
        if frf_freqs[0] <= f <= frf_freqs[-1]:
            ax_frf.axvline(f / 1000, color='r', linestyle='--', alpha=0.5, linewidth=0.7)
            ax_frf.text(f / 1000, ax_frf.get_ylim()[1] * 0.5,
                       f'  f{i+1}', fontsize=8, color='r')

    # Ultrasonic operating range
    ax_frf.axvspan(20, 40, alpha=0.1, color='green', label='Ultrasonic range (20-40 kHz)')
    ax_frf.legend(fontsize=9)

    # --- Mesh plot (bottom right) ---
    ax_mesh = fig.add_subplot(3, 3, 9)
    for elem in elements:
        poly = coords[np.append(elem, elem[0])]
        ax_mesh.plot(poly[:, 0] * 1000, poly[:, 1] * 1000, 'k-', linewidth=0.2)

    # Highlight fixed edge
    fixed_x = coords[coords[:, 0] < 1e-10]
    ax_mesh.plot(fixed_x[:, 0] * 1000, fixed_x[:, 1] * 1000,
                'rs', markersize=4, label='Fixed (hub)')

    # Highlight tip
    tip_node = np.argmax(coords[:, 0] + coords[:, 1])
    ax_mesh.plot(coords[tip_node, 0] * 1000, coords[tip_node, 1] * 1000,
                'b^', markersize=8, label='Force/Response point')

    ax_mesh.set_aspect('equal')
    ax_mesh.set_xlabel('x [mm]', fontsize=11)
    ax_mesh.set_ylabel('y [mm]', fontsize=11)
    ax_mesh.set_title(f'FE Mesh ({len(elements)} Q4 elements, {len(coords)} nodes)', fontsize=11)
    ax_mesh.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("FEM Modal Analysis — Ultrasonic Dicing Blade")
    print("Portfolio: DISCO Corporation")
    print("=" * 60)

    # Material
    D = constitutive_matrix(E, nu)
    print(f"\nMaterial: Steel (E={E/1e9:.0f} GPa, ν={nu}, ρ={rho} kg/m³)")
    print(f"Blade: {blade_length*1000:.0f} x {blade_height*1000:.0f} mm, t={thickness*1000:.1f} mm")

    # Mesh
    print(f"\nGenerating mesh ({nx}x{ny} = {nx*ny} Q4 elements)...")
    coords, elements, n_nodes = generate_mesh(blade_length, blade_height, nx, ny)
    n_dof = 2 * n_nodes
    print(f"  Nodes: {n_nodes}, DOFs: {n_dof}")

    # Assembly
    print("Assembling global matrices...")
    K, M = assemble(coords, elements, D, rho, thickness)

    # Boundary conditions
    fixed_dofs = get_fixed_dofs(coords, ny)
    print(f"  Fixed DOFs: {len(fixed_dofs)} (left edge clamped)")

    # Modal analysis
    n_modes = 8
    print(f"\nSolving eigenvalue problem for {n_modes} modes...")
    freqs_hz, modes = solve_modes(K, M, fixed_dofs, n_modes)

    print("\nNatural frequencies:")
    for i, f in enumerate(freqs_hz):
        print(f"  Mode {i+1}: {f:10.1f} Hz  ({f/1000:.2f} kHz)")

    # FRF
    print("\nComputing frequency response function (500 Hz - 50 kHz)...")
    frf_freqs, frf_data = compute_frf(K, M, fixed_dofs, coords,
                                       freq_range=(500, 50000), n_freq=600)

    # Visualization
    import os
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "vibration_modal_results.png")
    print("\nPlotting results...")
    plot_results(coords, elements, freqs_hz, modes, frf_freqs, frf_data, save_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
