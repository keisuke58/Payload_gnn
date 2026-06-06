# -*- coding: utf-8 -*-
"""
2D FEM Solver for Topology Optimization

Sigmund's 88-line topology optimization (MATLAB) の Python 実装。
MBB 梁ベンチマーク問題用。

Reference:
    Andreassen et al. (2011) "Efficient topology optimization in MATLAB
    using 88 lines of code" Struct Multidisc Optim 43:1-16

Unit: Arbitrary (normalized), Plane stress Q4 elements
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def element_stiffness(nu: float = 0.3) -> np.ndarray:
    """Compute 8x8 element stiffness matrix for unit-size Q4 plane stress element.

    Exact formulation from Andreassen et al. (2011), Eq. (1).
    """
    A11 = np.array([
        [12,  3, -6, -3],
        [ 3, 12,  3,  0],
        [-6,  3, 12, -3],
        [-3,  0, -3, 12]
    ], dtype=float)
    A12 = np.array([
        [-6, -3,  0,  3],
        [-3, -6, -3, -6],
        [ 0, -3, -6,  3],
        [ 3, -6,  3, -6]
    ], dtype=float)
    B11 = np.array([
        [-4,  3, -2,  9],
        [ 3, -4, -9,  4],
        [-2, -9, -4, -3],
        [ 9,  4, -3, -4]
    ], dtype=float)
    B12 = np.array([
        [ 2, -3,  4, -9],
        [-3,  2,  9, -2],
        [ 4,  9,  2,  3],
        [-9, -2,  3,  2]
    ], dtype=float)
    KE = 1.0 / (1.0 - nu**2) / 24.0 * (
        np.block([[A11, A12], [A12.T, A11]]) +
        nu * np.block([[B11, B12], [B12.T, B11]])
    )
    return KE


def density_filter(nelx: int, nely: int, rmin: float) -> tuple:
    """Prepare density filter (convolution-based) to prevent checkerboard patterns.

    Returns:
        H: filter matrix (sparse)
        Hs: row-sum of H for normalization
    """
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            imin = max(i - int(np.ceil(rmin)) + 1, 0)
            imax = min(i + int(np.ceil(rmin)), nelx)
            jmin = max(j - int(np.ceil(rmin)) + 1, 0)
            jmax = min(j + int(np.ceil(rmin)), nely)
            for ii in range(imin, imax):
                for jj in range(jmin, jmax):
                    col = ii * nely + jj
                    fac = rmin - np.sqrt((i - ii)**2 + (j - jj)**2)
                    if fac > 0:
                        if cc >= nfilter:
                            # Extend arrays if needed
                            iH = np.append(iH, np.zeros(nfilter, dtype=int))
                            jH = np.append(jH, np.zeros(nfilter, dtype=int))
                            sH = np.append(sH, np.zeros(nfilter))
                            nfilter *= 2
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = fac
                        cc += 1
    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])),
                    shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs


def solve_fem(nelx: int, nely: int, density: np.ndarray,
              penal: float = 3.0, E0: float = 1.0, Emin: float = 1e-9,
              nu: float = 0.3) -> tuple:
    """Solve 2D FEM for MBB beam (half model using symmetry).

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        density: Element densities, shape (nelx, nely)
        penal: SIMP penalization exponent
        E0: Young's modulus of solid material
        Emin: Young's modulus of void
        nu: Poisson's ratio

    Returns:
        compliance: Scalar compliance value
        U: Displacement vector
        ce: Element compliance, shape (nelx, nely)
    """
    KE = element_stiffness(nu)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Build global stiffness matrix using COO format
    edof_rows = []
    edof_cols = []
    edof_vals = []

    for elx in range(nelx):
        for ely in range(nely):
            # Node numbering (column-major like original 88-line code)
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([
                2*n1, 2*n1+1, 2*n2, 2*n2+1,
                2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
            ])
            # SIMP material interpolation
            xe = density[elx, ely]
            Ee = Emin + xe**penal * (E0 - Emin)
            ke = Ee * KE

            for i in range(8):
                for j in range(8):
                    edof_rows.append(edof[i])
                    edof_cols.append(edof[j])
                    edof_vals.append(ke[i, j])

    K = coo_matrix((edof_vals, (edof_rows, edof_cols)),
                    shape=(ndof, ndof)).tocsc()

    # Load vector: unit load at top-left corner (MBB half model)
    F = np.zeros(ndof)
    F[1] = -1.0  # downward force at node (0,0), y-dof

    # Boundary conditions (MBB half model):
    # - Left edge: fix x-displacement (symmetry)
    # - Bottom-right corner: fix y-displacement (roller support)
    fixed_x = np.arange(0, 2 * (nely + 1), 2)  # all left-edge x-dofs
    fixed_y = np.array([ndof - 1])  # bottom-right y-dof
    fixed_dofs = np.union1d(fixed_x, fixed_y)
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Solve
    U = np.zeros(ndof)
    U[free_dofs] = spsolve(K[free_dofs][:, free_dofs], F[free_dofs])

    # Element compliance
    ce = np.zeros((nelx, nely))
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([
                2*n1, 2*n1+1, 2*n2, 2*n2+1,
                2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
            ])
            Ue = U[edof]
            xe = density[elx, ely]
            Ee = Emin + xe**penal * (E0 - Emin)
            ce[elx, ely] = Ee * Ue @ KE @ Ue

    compliance = np.sum(ce)
    return compliance, U, ce


def optimality_criteria(nelx: int, nely: int, density: np.ndarray,
                        dc: np.ndarray, dv: np.ndarray,
                        volfrac: float, move: float = 0.2) -> np.ndarray:
    """Optimality Criteria (OC) update for topology optimization.

    Args:
        nelx, nely: mesh dimensions
        density: current densities (nelx, nely)
        dc: sensitivity of compliance (nelx, nely)
        dv: sensitivity of volume (nelx, nely)
        volfrac: target volume fraction
        move: maximum density change per iteration

    Returns:
        xnew: updated densities (nelx, nely)
    """
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        # Avoid division by zero / negative sqrt
        ratio = np.where(dv * lmid != 0, -dc / (dv * lmid), 1.0)
        ratio = np.maximum(ratio, 0.0)
        xnew = np.maximum(0.001,
                np.maximum(density - move,
                np.minimum(1.0,
                np.minimum(density + move,
                           density * np.sqrt(ratio)))))
        if xnew.sum() > volfrac * nelx * nely:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def topology_optimization_oc(nelx: int = 60, nely: int = 20,
                              volfrac: float = 0.5, penal: float = 3.0,
                              rmin: float = 1.5, max_iter: int = 200,
                              tol: float = 0.01) -> tuple:
    """Run standard OC-based topology optimization (baseline).

    Returns:
        density: optimized density field (nelx, nely)
        history: list of (iteration, compliance, volume_fraction, change)
    """
    print(f"OC Topology Optimization: {nelx}x{nely}, vf={volfrac}, penal={penal}")
    density = np.full((nelx, nely), volfrac)
    H, Hs = density_filter(nelx, nely, rmin)

    history = []
    for it in range(max_iter):
        # FEM solve
        compliance, U, ce = solve_fem(nelx, nely, density, penal)

        # Sensitivities
        dc = -penal * density**(penal - 1) * ce
        dv = np.ones((nelx, nely))

        # Filter sensitivities
        dc_flat = dc.flatten()
        dv_flat = dv.flatten()
        dc_filtered = np.array(H.dot(dc_flat * density.flatten()) / Hs / np.maximum(0.001, density.flatten()))
        dv_filtered = np.array(H.dot(dv_flat) / Hs)
        dc = dc_filtered.reshape(nelx, nely)
        dv = dv_filtered.reshape(nelx, nely)

        # OC update
        density_new = optimality_criteria(nelx, nely, density, dc, dv, volfrac)
        change = np.max(np.abs(density_new - density))
        density = density_new

        vf = density.sum() / (nelx * nely)
        history.append((it, compliance, vf, change))

        if it % 10 == 0 or change < tol:
            print(f"  Iter {it:4d}: C={compliance:.4f}, Vf={vf:.4f}, Change={change:.4f}")

        if change < tol and it > 10:
            print(f"  Converged at iteration {it}")
            break

    return density, history


def plot_topology(density: np.ndarray, title: str = "Topology Optimization Result",
                  save_path: str = None):
    """Plot density field."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(-density.T, cmap='gray', interpolation='none',
              extent=[0, density.shape[0], 0, density.shape[1]])
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ==============================================================================
# Main: Run OC baseline for comparison
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("2D FEM Solver + OC Topology Optimization (Baseline)")
    print("MBB Beam Benchmark")
    print("=" * 60)

    # Run OC baseline
    density, history = topology_optimization_oc(
        nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5
    )

    # Plot result
    plot_topology(density, "OC Baseline: MBB Beam (60x20, vf=0.5)",
                  save_path="results/oc_baseline_mbb.png")

    # Print final stats
    final = history[-1]
    print(f"\nFinal: C={final[1]:.4f}, Vf={final[2]:.4f}, "
          f"Iterations={final[0]+1}")
