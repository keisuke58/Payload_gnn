# -*- coding: utf-8 -*-
# jaxfem_fairing.py
# JAX-FEM prototype for H3 fairing (differentiable FEM)
#
# Usage:
#   pip install jax jaxlib jax-fem
#   python src/jaxfem_fairing.py [--output output/]
#
# Benefits: Differentiable w.r.t. material/geometry -> inverse problems, PINN integration

import argparse
import os
import sys

# H3 dimensions (mm)
RADIUS = 2600.0
H_BARREL = 5000.0
CORE_T = 38.0
E_CFRP = 70000.0
NU_CFRP = 0.3


def run_jaxfem_plate(output_dir: str, nx: int = 10, ny: int = 20) -> dict:
    """
    JAX-FEM linear elasticity on rectangular plate.
    Ref: https://github.com/deepmodeling/jax-fem
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("ERROR: jax not installed. pip install jax jaxlib")
        return {"error": "jax not installed"}

    try:
        from jax_fem.problem import Problem
        from jax_fem.linear_elasticity import LinearElasticity
        from jax_fem import mesh
    except ImportError:
        print("WARNING: jax-fem not found. pip install jax-fem")
        print("  Falling back to minimal JAX-only prototype (no FEM solve)")
        return run_minimal_jax_prototype(output_dir, nx, ny)

    # Create mesh (simplified: jax-fem mesh format)
    # jax-fem typically uses external mesh; we create a simple grid
    def create_simple_mesh(nx, ny, Lx=500.0, Ly=1000.0):
        x = jnp.linspace(0, Lx, nx + 1)
        y = jnp.linspace(0, Ly, ny + 1)
        points = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append([x[i], y[j]])
        points = jnp.array(points)
        # Triangulate (Delaunay-like)
        cells = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nx + 1)
                n3 = n2 + 1
                cells.append([n0, n1, n2])
                cells.append([n1, n3, n2])
        cells = jnp.array(cells)
        return points, cells

    points, cells = create_simple_mesh(nx, ny)
    mesh_data = (points, cells)

    def dirichlet_val(point):
        return 0.0

    def get_dirichlet_location(point):
        return jnp.isclose(point[1], 0.0)

    problem = LinearElasticity(
        mesh_data,
        vec=2,
        dim=2,
        dirichlet_bc_info=[{"axis": 1, "location": get_dirichlet_location, "value": dirichlet_val}],
    )

    # Material
    E = 70000.0
    nu = 0.3
    problem.set_params([E, nu])

    # Solve
    sol = problem.solve()

    os.makedirs(output_dir, exist_ok=True)
    import numpy as np
    pts = np.array(points)
    u = np.array(sol)
    nodes_path = os.path.join(output_dir, "nodes.csv")
    with open(nodes_path, "w") as f:
        f.write("node_id,x,y,z,ux,uy,uz,temp\n")
        for i in range(len(pts)):
            ux = u[i, 0] if u.ndim > 1 else u[i]
            uy = u[i, 1] if u.shape[-1] > 1 else 0.0
            f.write("%d,%.6e,%.6e,0,%.6e,%.6e,0,20\n" % (i, pts[i, 0], pts[i, 1], ux, uy))

    return {"n_nodes": len(pts), "nodes_csv": nodes_path, "solver": "jax-fem"}


def run_minimal_jax_prototype(output_dir: str, nx: int = 10, ny: int = 20) -> dict:
    """
    Minimal JAX prototype: no jax-fem, just demonstrates differentiable
    forward model stub for PINN / inverse problem integration.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    # Stiffness matrix assembly (simplified 1D bar)
    def assemble_K(E, A, L, n):
        k_local = E * A / L * jnp.array([[1, -1], [-1, 1]])
        K = jnp.zeros((n + 1, n + 1))
        for i in range(n):
            K = K.at[i : i + 2, i : i + 2].add(k_local)
        return K

    def solve_bar(E, A, L, n, F):
        K = assemble_K(E, A, L, n)
        K_bc = K[1:, 1:]
        F_bc = F[1:]
        u = jnp.linalg.solve(K_bc, F_bc)
        return jnp.concatenate([jnp.array([0.0]), u])

    # Differentiable: d(u)/d(E)
    n = 20
    E_val = 70000.0
    A_val = 1.0
    L_val = 1000.0
    F_val = jnp.zeros(n + 1)
    F_val = F_val.at[-1].set(100.0)

    u = solve_bar(E_val, A_val, L_val, n, F_val)
    jac_E = jax.jacfwd(lambda E: solve_bar(E, A_val, L_val, n, F_val))(E_val)
    print("JAX prototype: u[-1]=%.6e, du/dE shape=%s" % (float(u[-1]), jac_E.shape))

    os.makedirs(output_dir, exist_ok=True)
    nodes_path = os.path.join(output_dir, "nodes.csv")
    x = np.linspace(0, L_val, n + 1)
    with open(nodes_path, "w") as f:
        f.write("node_id,x,y,z,ux,uy,uz,temp\n")
        for i in range(len(x)):
            f.write("%d,%.6e,0,0,%.6e,0,0,20\n" % (i, x[i], float(u[i])))

    return {
        "n_nodes": n + 1,
        "nodes_csv": nodes_path,
        "solver": "jax-minimal",
        "differentiable": True,
    }


def main():
    parser = argparse.ArgumentParser(description="JAX-FEM fairing prototype")
    parser.add_argument("--output", "-o", default="dataset_output/jaxfem_proto")
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=20)
    args = parser.parse_args()

    result = run_jaxfem_plate(args.output, args.nx, args.ny)
    print("JAX-FEM prototype done: %s" % result)
    return result


if __name__ == "__main__":
    main()
