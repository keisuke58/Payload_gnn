# -*- coding: utf-8 -*-
# fenics_fairing.py
# FEniCS/dolfinx prototype for H3 fairing FEM (simplified 2D axisymmetric or 3D)
#
# Usage:
#   pip install dolfinx meshio pygmsh  # or: conda install -c conda-forge fenics-dolfinx
#   python src/fenics_fairing.py [--2d] [--output output/]
#
# Output: CSV nodes.csv, elements.csv compatible with build_graph.py

import argparse
import math
import os
import sys

# H3 Type-S dimensions (mm)
RADIUS = 2600.0
H_BARREL = 5000.0
H_NOSE = 5400.0
TOTAL_HEIGHT = H_BARREL + H_NOSE
CORE_T = 38.0
FACE_T = 1.0

# Material (CFRP simplified as isotropic for prototype)
E_CFRP = 70000.0  # MPa (effective)
NU_CFRP = 0.3
E_CORE = 1000.0   # MPa (out-of-plane dominant)
NU_CORE = 0.01


def run_2d_axisymmetric(output_dir: str, n_cells: int = 100) -> dict:
    """2D axisymmetric linear elasticity (cylinder section)."""
    try:
        from dolfinx import fem, mesh, log
        from dolfinx.fem import Function, FunctionSpace, dirichletbc
        from dolfinx.fem.petsc import LinearProblem
        from dolfinx.io import XDMFFile
        from mpi4py import MPI
        import numpy as np
        import ufl
    except ImportError as e:
        print("ERROR: dolfinx not installed. Run: pip install fenics-dolfinx")
        print("  Or: conda install -c conda-forge fenics-dolfinx")
        raise e

    # Create 1D mesh along r (radial) for axisymmetric
    # Simplified: single layer cylinder slice
    domain = mesh.create_interval(MPI.COMM_WORLD, n_cells, (RADIUS, RADIUS + CORE_T + FACE_T))
    V = fem.functionspace(domain, ("Lagrange", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E = fem.Constant(domain, E_CFRP)
    nu = fem.Constant(domain, NU_CFRP)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Axisymmetric stress (simplified 1D radial)
    r = ufl.SpatialCoordinate(domain)[0]
    eps_rr = ufl.dx(u) / ufl.dx(ufl.SpatialCoordinate(domain)[0])
    eps_tt = u / r  # hoop
    eps = ufl.as_vector([eps_rr, eps_tt])
    sigma_rr = lmbda * (eps_rr + eps_tt) + 2 * mu * eps_rr
    sigma_tt = lmbda * (eps_rr + eps_tt) + 2 * mu * eps_tt

    # Pressure load at outer surface
    p_ext = fem.Constant(domain, 0.05)  # 50 kPa
    F = ufl.inner(sigma_rr, v) * ufl.dx - p_ext * v * ufl.ds

    # BC: fixed at inner radius
    def inner_boundary(x):
        return np.isclose(x[0], RADIUS)

    fd = mesh.locate_entities_boundary(domain, 0, inner_boundary)
    bc = dirichletbc(fem.Constant(domain, 0.0), fem.locate_dofs_topological(V, 0, fd), V)

    problem = LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Export
    os.makedirs(output_dir, exist_ok=True)
    with XDMFFile(domain.comm, os.path.join(output_dir, "displacement.xdmf"), "w") as f:
        f.write_mesh(domain)
        uh.name = "u"
        f.write_function(uh)

    # Extract nodes for CSV (1D -> expand to 2D cylindrical for compatibility)
    x = domain.geometry.x
    u_vals = uh.x.array
    nodes = []
    for i in range(len(x)):
        r_val = x[i, 0]
        nodes.append([i, r_val, 0.0, 0.0, u_vals[i] if i < len(u_vals) else 0.0, 0.0, 0.0, 20.0])

    return {"n_nodes": len(nodes), "solver": "dolfinx-2d-axisymmetric"}


def run_3d_quarter(output_dir: str, mesh_size: float = 100.0) -> dict:
    """3D quarter cylinder (90° arc) with pygmsh + dolfinx."""
    try:
        import pygmsh
        import meshio
        import numpy as np
    except ImportError:
        print("WARNING: pygmsh/meshio not found. Install: pip install pygmsh meshio")
        return run_2d_axisymmetric(output_dir, n_cells=50)

    try:
        from dolfinx import fem, mesh as dmesh
        from dolfinx.io import gmshio
        from mpi4py import MPI
        import ufl
    except ImportError:
        print("Falling back to 2D axisymmetric (dolfinx required for 3D)")
        return run_2d_axisymmetric(output_dir, n_cells=50)

    # Build geometry with pygmsh: quarter cylinder
    with pygmsh.geo.Geometry() as geom:
        cyl = geom.add_cylinder(
            [0, 0, 0], [0, H_BARREL, 0], RADIUS + CORE_T + FACE_T,
            mesh_size=mesh_size
        )
        # Cut to quarter (0-90 deg in XZ)
        box = geom.add_box([0, -1, 0], [RADIUS + 2000, H_BARREL + 2, RADIUS + 2000])
        quarter = geom.boolean_difference([cyl], [box])
        geom.generate_mesh(dim=3, algorithm=5)

    msh_path = os.path.join(output_dir, "fairing_quarter.msh")
    geom.mesh.write(msh_path)

    # Convert to dolfinx
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(
        geom.model, MPI.COMM_WORLD, 0, gdim=3
    )
    # ... solve (abbreviated for prototype)
    return {"n_nodes": 0, "solver": "dolfinx-3d-quarter", "msh": msh_path}


def run_simplified_plate(output_dir: str, nx: int = 20, ny: int = 40) -> dict:
    """
    Simplified flat plate (curved fairing approximated as flat).
    Easiest to run, good for pipeline validation.
    """
    try:
        from dolfinx import fem, mesh
        from dolfinx.fem import dirichletbc
        from dolfinx.fem.petsc import LinearProblem
        from mpi4py import MPI
        import numpy as np
        import ufl
    except ImportError as e:
        print("ERROR: dolfinx required. pip install fenics-dolfinx")
        raise e

    # Rectangular domain: Lx x Ly (mm)
    Lx = 1000.0   # circumferential slice
    Ly = 2000.0   # axial
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0, 0], [Lx, Ly]],
        [nx, ny],
        mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E = fem.Constant(domain, E_CFRP)
    nu = fem.Constant(domain, NU_CFRP)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(2) + 2 * mu * eps(u)

    # Pressure load
    p = fem.Constant(domain, 0.05)
    f = ufl.as_vector([0.0, 0.0])
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx  # + traction on boundary if needed

    # BC: clamp bottom
    def bottom(x):
        return np.isclose(x[1], 0.0)

    fd = mesh.locate_entities_boundary(domain, 1, bottom)
    dofs = fem.locate_dofs_topological(V, 1, fd)
    bc = dirichletbc(np.array([0.0, 0.0], dtype=np.float64), dofs, V)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Export to CSV (compatible with build_graph)
    os.makedirs(output_dir, exist_ok=True)
    x = domain.geometry.x
    u_arr = uh.x.array.reshape(-1, 2)
    nodes_path = os.path.join(output_dir, "nodes.csv")
    with open(nodes_path, "w") as f:
        f.write("node_id,x,y,z,ux,uy,uz,temp\n")
        for i in range(len(x)):
            ux, uy = u_arr[i, 0], u_arr[i, 1] if u_arr.shape[1] > 1 else 0.0
            f.write("%d,%.6e,%.6e,0,%.6e,%.6e,0,20\n" % (i, x[i, 0], x[i, 1], ux, uy))

    # Elements
    t = domain.topology
    t.create_connectivity(2, 0)
    conn = t.connectivity(2, 0)
    elems_path = os.path.join(output_dir, "elements.csv")
    with open(elems_path, "w") as f:
        f.write("elem_id,elem_type,n1,n2,n3,n4,area\n")
        for e in range(conn.num_nodes):
            nodes_e = conn.links(e)
            n_str = ",".join(str(n) for n in nodes_e)
            f.write("%d,TRI3,%s,1\n" % (e, n_str))

    return {
        "n_nodes": len(x),
        "n_elements": conn.num_nodes,
        "nodes_csv": nodes_path,
        "elements_csv": elems_path,
        "solver": "dolfinx-plate",
    }


def main():
    parser = argparse.ArgumentParser(description="FEniCS fairing FEM prototype")
    parser.add_argument("--2d", action="store_true", help="2D axisymmetric (default: plate)")
    parser.add_argument("--3d", action="store_true", help="3D quarter cylinder (experimental)")
    parser.add_argument("--output", "-o", default="dataset_output/fenics_proto", help="Output dir")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=40)
    args = parser.parse_args()

    if getattr(args, 'three_d', False):
        result = run_3d_quarter(args.output)
    elif getattr(args, 'two_d', False):
        result = run_2d_axisymmetric(args.output)
    else:
        result = run_simplified_plate(args.output, args.nx, args.ny)

    print("FEniCS prototype done: %s" % result)
    return result


if __name__ == "__main__":
    main()
