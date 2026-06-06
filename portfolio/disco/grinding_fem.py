#!/usr/bin/env python3
"""
FEM Thermal-Structural Coupled Analysis of Wafer Grinding
==========================================================
Portfolio: DISCO Corporation — Precision Grinding Equipment

Physics:
- Transient heat conduction with moving Gaussian heat source (grinding wheel)
- Convection cooling (coolant) on top surface
- Thermal stress analysis (plane stress, thermal expansion)
- Silicon wafer material properties

FEM Implementation:
- Q4 isoparametric elements (bilinear quadrilateral)
- Gauss quadrature (2x2)
- Sparse assembly (scipy COO → CSR)
- Backward Euler time integration for thermal
- Direct sparse solver

Author: Nishioka
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os
import time

# ============================================================
# Material properties — Silicon wafer
# ============================================================
E = 130e9          # Young's modulus [Pa]
nu = 0.28          # Poisson's ratio
alpha_th = 2.6e-6  # Thermal expansion coefficient [1/K]
k_th = 130.0       # Thermal conductivity [W/m·K]
rho = 2330.0       # Density [kg/m³]
cp = 700.0         # Specific heat [J/kg·K]
T_ref = 300.0      # Reference temperature [K] (no thermal stress)

# ============================================================
# Geometry and mesh
# ============================================================
wafer_length = 10e-3    # 10 mm (grinding contact zone)
wafer_thickness = 0.5e-3  # 0.5 mm (thinned wafer)

nx_elem = 80    # elements along length
ny_elem = 12    # elements along thickness

# ============================================================
# Grinding parameters
# ============================================================
heat_flux_peak = 5e7    # Peak heat flux [W/m²]
heat_source_width = 0.5e-3   # Gaussian width [m]
grinding_speed = 2e-3   # Wheel speed across wafer [m/s]
h_conv = 5000.0         # Convection coefficient [W/m²·K] (coolant)
T_coolant = 300.0       # Coolant temperature [K]
t_total = 3e-3          # Total simulation time [s]
n_time_steps = 30       # Number of time steps

# ============================================================
# Q4 Element functions
# ============================================================
def shape_functions(xi, eta):
    """Bilinear Q4 shape functions and derivatives."""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    dN = 0.25 * np.array([
        [-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)],
        [-(1 - xi),  -(1 + xi),   (1 + xi),   (1 - xi)]
    ])
    return N, dN

# 2x2 Gauss quadrature
gp = 1.0 / np.sqrt(3.0)
gauss_pts = np.array([-gp, gp])
gauss_wts = np.array([1.0, 1.0])

# ============================================================
# Mesh generation
# ============================================================
def generate_mesh(Lx, Ly, nx, ny):
    """Structured Q4 mesh."""
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
    return coords, np.array(elements), n_nodes

# ============================================================
# Thermal analysis
# ============================================================
def assemble_thermal(coords, elements, k_th, rho, cp):
    """Assemble thermal conductivity [K_th] and capacity [C_th] matrices."""
    n_nodes = coords.shape[0]
    rows, cols, kvals, cvals = [], [], [], []

    for elem in elements:
        ce = coords[elem]
        Ke = np.zeros((4, 4))
        Ce = np.zeros((4, 4))

        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):
                w = gauss_wts[i] * gauss_wts[j]
                N, dN = shape_functions(xi, eta)
                J = dN @ ce
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN  # (2x4)

                # Conductivity: ∫ k ∇N · ∇N dA
                Ke += k_th * (dN_dx.T @ dN_dx) * detJ * w

                # Capacity: ∫ ρ cp N N^T dA
                Ce += rho * cp * np.outer(N, N) * detJ * w

        for a in range(4):
            for b in range(4):
                rows.append(elem[a])
                cols.append(elem[b])
                kvals.append(Ke[a, b])
                cvals.append(Ce[a, b])

    K_th = sparse.coo_matrix((kvals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    C_th = sparse.coo_matrix((cvals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return K_th, C_th

def apply_convection(K_th, coords, elements, ny_elem, h_conv, T_coolant, n_nodes):
    """Add convection BC on top surface: h*(T - T_inf)."""
    # Top surface: y = wafer_thickness
    tol = 1e-10
    top_nodes = np.where(np.abs(coords[:, 1] - coords[:, 1].max()) < tol)[0]

    # Build convection matrix and load vector
    rows, cols, hvals = [], [], []
    f_conv = np.zeros(n_nodes)

    # Find top surface edges
    for elem in elements:
        ce = coords[elem]
        # Check if nodes 2,3 (top of Q4) are on top surface
        if elem[2] in top_nodes and elem[3] in top_nodes:
            # Edge between nodes 2 and 3
            L = np.linalg.norm(ce[2] - ce[3])
            # Consistent boundary: ∫ h N_i N_j dS
            he = h_conv * L / 6.0 * np.array([[2, 1], [1, 2]])
            edge_nodes = [elem[2], elem[3]]
            for a in range(2):
                f_conv[edge_nodes[a]] += h_conv * T_coolant * L / 2.0
                for b in range(2):
                    rows.append(edge_nodes[a])
                    cols.append(edge_nodes[b])
                    hvals.append(he[a, b])

    H = sparse.coo_matrix((hvals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return K_th + H, f_conv

def heat_source_vector(coords, elements, t, heat_flux_peak, speed, width):
    """Moving Gaussian heat source on top surface."""
    n_nodes = coords.shape[0]
    f = np.zeros(n_nodes)
    x_center = speed * t  # Heat source position

    tol = 1e-10
    y_max = coords[:, 1].max()

    for elem in elements:
        ce = coords[elem]
        # Top edge: nodes 2 and 3
        if abs(ce[2, 1] - y_max) < tol and abs(ce[3, 1] - y_max) < tol:
            edge_nodes = [elem[2], elem[3]]
            L = np.linalg.norm(ce[2] - ce[3])
            # 2-point Gauss on edge
            for gp_1d, wt_1d in zip([-1/np.sqrt(3), 1/np.sqrt(3)], [1.0, 1.0]):
                # Physical x on edge
                x_phys = 0.5 * (ce[3, 0] * (1 - gp_1d) + ce[2, 0] * (1 + gp_1d))
                # Gaussian heat flux
                q = heat_flux_peak * np.exp(-((x_phys - x_center) / width)**2)
                # Shape functions on edge (linear)
                N1 = 0.5 * (1 - gp_1d)  # node 3
                N2 = 0.5 * (1 + gp_1d)  # node 2
                f[edge_nodes[1]] += q * N1 * L/2 * wt_1d  # node 3
                f[edge_nodes[0]] += q * N2 * L/2 * wt_1d  # node 2

    return f

def solve_thermal_transient(coords, elements, K_th_conv, C_th, f_conv,
                             n_nodes, dt, n_steps):
    """Backward Euler: (C/dt + K) T^{n+1} = C/dt T^n + f^{n+1}."""
    T = np.full(n_nodes, T_ref)
    T_history = [T.copy()]

    A = C_th / dt + K_th_conv

    for step in range(n_steps):
        t = (step + 1) * dt
        f_heat = heat_source_vector(coords, elements, t,
                                     heat_flux_peak, grinding_speed,
                                     heat_source_width)
        rhs = C_th / dt @ T + f_heat + f_conv
        T = spsolve(A, rhs)
        T_history.append(T.copy())

    return T, T_history

# ============================================================
# Structural analysis — Thermal stress
# ============================================================
def plane_stress_D(E, nu):
    """Plane stress constitutive matrix."""
    c = E / (1 - nu**2)
    return c * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1-nu)/2]
    ])

def assemble_structural(coords, elements, D_mat):
    """Assemble structural stiffness matrix."""
    n_dof = 2 * coords.shape[0]
    rows, cols, kvals = [], [], []

    for elem in elements:
        ce = coords[elem]
        Ke = np.zeros((8, 8))

        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):
                w = gauss_wts[i] * gauss_wts[j]
                N, dN = shape_functions(xi, eta)
                J = dN @ ce
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN

                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2*k]     = dN_dx[0, k]
                    B[1, 2*k+1]   = dN_dx[1, k]
                    B[2, 2*k]     = dN_dx[1, k]
                    B[2, 2*k+1]   = dN_dx[0, k]

                Ke += (B.T @ D_mat @ B) * detJ * w

        dof_map = np.zeros(8, dtype=int)
        for ii, node in enumerate(elem):
            dof_map[2*ii] = 2*node
            dof_map[2*ii+1] = 2*node + 1

        for a in range(8):
            for b in range(8):
                rows.append(dof_map[a])
                cols.append(dof_map[b])
                kvals.append(Ke[a, b])

    return sparse.coo_matrix((kvals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

def thermal_load_vector(coords, elements, T_field, D_mat, alpha_th, T_ref):
    """Compute nodal force from thermal strain: ε_th = α ΔT [1, 1, 0]^T."""
    n_dof = 2 * coords.shape[0]
    f_th = np.zeros(n_dof)

    for elem in elements:
        ce = coords[elem]
        fe = np.zeros(8)

        for i, xi in enumerate(gauss_pts):
            for j, eta in enumerate(gauss_pts):
                w = gauss_wts[i] * gauss_wts[j]
                N, dN = shape_functions(xi, eta)
                J = dN @ ce
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = invJ @ dN

                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2*k]     = dN_dx[0, k]
                    B[1, 2*k+1]   = dN_dx[1, k]
                    B[2, 2*k]     = dN_dx[1, k]
                    B[2, 2*k+1]   = dN_dx[0, k]

                # Temperature at Gauss point
                T_gp = N @ T_field[elem]
                dT = T_gp - T_ref

                # Thermal strain
                eps_th = alpha_th * dT * np.array([1.0, 1.0, 0.0])

                # Thermal stress
                sigma_th = D_mat @ eps_th

                # Equivalent nodal force
                fe += (B.T @ sigma_th) * detJ * w

        dof_map = np.zeros(8, dtype=int)
        for ii, node in enumerate(elem):
            dof_map[2*ii] = 2*node
            dof_map[2*ii+1] = 2*node + 1
        f_th[dof_map] += fe

    return f_th

def solve_structural(K_s, f_th, coords):
    """Solve K u = f with boundary conditions."""
    n_dof = K_s.shape[0]
    tol = 1e-10

    # BC: fix left edge (x=0) in x, fix bottom-left corner in y
    fixed_dofs = []
    for i, (x, y) in enumerate(coords):
        if x < tol:
            fixed_dofs.append(2*i)  # fix u_x
        if x < tol and y < tol:
            fixed_dofs.append(2*i + 1)  # fix u_y at corner

    fixed_dofs = np.array(fixed_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    u = np.zeros(n_dof)
    K_ff = K_s[np.ix_(free_dofs, free_dofs)]
    f_ff = f_th[free_dofs]
    u[free_dofs] = spsolve(K_ff, f_ff)
    return u

def compute_von_mises(coords, elements, u, T_field, D_mat, alpha_th, T_ref):
    """Compute von Mises stress at element centers."""
    n_elem = len(elements)
    vm_stress = np.zeros(n_elem)
    elem_centers = np.zeros((n_elem, 2))

    for e_idx, elem in enumerate(elements):
        ce = coords[elem]
        elem_centers[e_idx] = ce.mean(axis=0)

        # Evaluate at element center (xi=0, eta=0)
        N, dN = shape_functions(0.0, 0.0)
        J = dN @ ce
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN

        B = np.zeros((3, 8))
        for k in range(4):
            B[0, 2*k]     = dN_dx[0, k]
            B[1, 2*k+1]   = dN_dx[1, k]
            B[2, 2*k]     = dN_dx[1, k]
            B[2, 2*k+1]   = dN_dx[0, k]

        ue = np.zeros(8)
        for ii, node in enumerate(elem):
            ue[2*ii] = u[2*node]
            ue[2*ii+1] = u[2*node+1]

        # Mechanical strain = total - thermal
        eps_total = B @ ue
        T_gp = N @ T_field[elem]
        eps_th = alpha_th * (T_gp - T_ref) * np.array([1.0, 1.0, 0.0])
        eps_mech = eps_total - eps_th

        sigma = D_mat @ eps_mech
        sx, sy, txy = sigma
        vm = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)
        vm_stress[e_idx] = vm

    return vm_stress, elem_centers

# ============================================================
# Visualization
# ============================================================
def plot_results(coords, elements, T_field, u, vm_stress, elem_centers, save_path):
    """4-panel figure: temperature, von Mises, deformation, surface profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('FEM Thermal-Structural Coupled Analysis — Wafer Grinding (DISCO)',
                 fontsize=15, fontweight='bold')

    # Triangulation for node-based plots
    triangles = []
    for elem in elements:
        triangles.append([elem[0], elem[1], elem[2]])
        triangles.append([elem[0], elem[2], elem[3]])
    tri = mtri.Triangulation(coords[:, 0] * 1000, coords[:, 1] * 1000, triangles)

    # (a) Temperature distribution
    ax = axes[0, 0]
    tcf = ax.tripcolor(tri, T_field - 273.15, shading='gouraud', cmap='hot')
    plt.colorbar(tcf, ax=ax, label='Temperature [°C]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('(a) Temperature Distribution')
    ax.set_aspect('equal')

    # Show heat source position
    t_final = t_total
    x_source = grinding_speed * t_final * 1000
    ax.annotate('Grinding\nwheel', xy=(x_source, wafer_thickness*1000),
                xytext=(x_source, wafer_thickness*1000 + 0.15),
                fontsize=9, ha='center', color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # (b) Von Mises stress
    ax = axes[0, 1]
    sc = ax.scatter(elem_centers[:, 0] * 1000, elem_centers[:, 1] * 1000,
                    c=vm_stress / 1e6, s=8, cmap='jet', marker='s')
    plt.colorbar(sc, ax=ax, label='Von Mises Stress [MPa]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('(b) Von Mises Stress Distribution')
    ax.set_aspect('equal')

    # (c) Deformed shape (magnified)
    ax = axes[1, 0]
    ux = u[0::2]
    uy = u[1::2]
    max_disp = max(np.abs(u).max(), 1e-15)
    scale = wafer_length * 0.05 / max_disp

    deformed_x = (coords[:, 0] + ux * scale) * 1000
    deformed_y = (coords[:, 1] + uy * scale) * 1000
    tri_def = mtri.Triangulation(deformed_x, deformed_y, triangles)
    disp_mag = np.sqrt(ux**2 + uy**2) * 1e6  # [μm]
    tcf = ax.tripcolor(tri_def, disp_mag, shading='gouraud', cmap='viridis')
    plt.colorbar(tcf, ax=ax, label='Displacement [μm]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'(c) Deformed Shape (×{scale:.0f} magnified)')
    ax.set_aspect('equal')

    # (d) Surface profiles
    ax = axes[1, 1]
    tol = 1e-10
    y_max = coords[:, 1].max()
    top_mask = np.abs(coords[:, 1] - y_max) < tol
    top_x = coords[top_mask, 0] * 1000
    top_T = T_field[top_mask] - 273.15
    sort_idx = np.argsort(top_x)

    ax.plot(top_x[sort_idx], top_T[sort_idx], 'r-', linewidth=2, label='Temperature [°C]')
    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', color='r', fontsize=11)
    ax.tick_params(axis='y', labelcolor='r')

    # Von Mises on top surface (nearest elements)
    ax2 = ax.twinx()
    top_elem_mask = np.abs(elem_centers[:, 1] - y_max) < (wafer_thickness / ny_elem * 0.6)
    if top_elem_mask.any():
        top_ec_x = elem_centers[top_elem_mask, 0] * 1000
        top_vm = vm_stress[top_elem_mask] / 1e6
        sort_idx2 = np.argsort(top_ec_x)
        ax2.plot(top_ec_x[sort_idx2], top_vm[sort_idx2], 'b-', linewidth=2,
                 label='Von Mises [MPa]')
    ax2.set_ylabel('Von Mises Stress [MPa]', color='b', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='b')

    ax.set_title('(d) Top Surface Profiles')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

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
    print("FEM Thermal-Structural Analysis — Wafer Grinding")
    print("Portfolio: DISCO Corporation")
    print("=" * 60)

    print(f"\nMaterial: Silicon (E={E/1e9:.0f} GPa, ν={nu}, α={alpha_th*1e6:.1f} μ/K)")
    print(f"Wafer: {wafer_length*1000:.1f} x {wafer_thickness*1000:.2f} mm")
    print(f"Grinding: v={grinding_speed*1000:.0f} mm/s, q_peak={heat_flux_peak:.1e} W/m²")

    # Mesh
    print(f"\nMesh: {nx_elem}x{ny_elem} = {nx_elem*ny_elem} Q4 elements")
    coords, elements, n_nodes = generate_mesh(wafer_length, wafer_thickness,
                                                nx_elem, ny_elem)
    print(f"  Nodes: {n_nodes}")

    # Thermal analysis
    print("\n--- Thermal Analysis ---")
    dt = t_total / n_time_steps
    print(f"Time stepping: dt={dt*1e3:.3f} ms, {n_time_steps} steps")

    print("Assembling thermal matrices...")
    K_th, C_th = assemble_thermal(coords, elements, k_th, rho, cp)

    print("Applying convection BC (h={:.0f} W/m²K)...".format(h_conv))
    K_th_conv, f_conv = apply_convection(K_th, coords, elements, ny_elem,
                                          h_conv, T_coolant, n_nodes)

    print("Solving transient thermal...")
    T_field, T_history = solve_thermal_transient(coords, elements, K_th_conv, C_th,
                                                  f_conv, n_nodes, dt, n_time_steps)
    T_max = T_field.max() - 273.15
    print(f"  Max temperature: {T_max:.1f} °C")

    # Structural analysis
    print("\n--- Structural Analysis ---")
    D_mat = plane_stress_D(E, nu)
    print("Assembling structural stiffness...")
    K_s = assemble_structural(coords, elements, D_mat)

    print("Computing thermal load vector...")
    f_th = thermal_load_vector(coords, elements, T_field, D_mat, alpha_th, T_ref)

    print("Solving structural system...")
    u = solve_structural(K_s, f_th, coords)
    u_max = np.abs(u).max() * 1e6
    print(f"  Max displacement: {u_max:.3f} μm")

    print("Computing von Mises stress...")
    vm_stress, elem_centers = compute_von_mises(coords, elements, u, T_field,
                                                 D_mat, alpha_th, T_ref)
    print(f"  Max von Mises: {vm_stress.max()/1e6:.1f} MPa")
    print(f"  Si fracture strength: ~160 MPa (safe)" if vm_stress.max()/1e6 < 160
          else f"  WARNING: exceeds Si fracture strength (~160 MPa)")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "grinding_fem_results.png")
    print("\nPlotting results...")
    plot_results(coords, elements, T_field, u, vm_stress, elem_centers, save_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
