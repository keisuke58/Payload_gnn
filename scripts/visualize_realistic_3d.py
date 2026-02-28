#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beautiful 3D visualization of Realistic H3 Fairing FEM results.
Generates publication-quality mesh, stress, displacement, and temperature views
using PyVista for Phase 1 and Phase 2 realistic fairing models.
"""
import os
import sys
import numpy as np
import pandas as pd

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
import pyvista as pv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_DIR = os.path.join(PROJECT_ROOT, 'dataset_realistic', 'phase1')
PHASE2_DIR = os.path.join(PROJECT_ROOT, 'dataset_realistic', 'phase2')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures', 'realistic_fairing_3d')
os.makedirs(FIG_DIR, exist_ok=True)

# H3 fairing geometry constants
RADIUS = 2600.0
CORE_T = 38.0
OGIVE_H = 5400.0
BARREL_H = 5000.0
TOTAL_H = BARREL_H + OGIVE_H
OGIVE_R = (RADIUS**2 + OGIVE_H**2) / (2 * RADIUS)

# Opening definitions (Phase 2)
OPENINGS = {
    'AccessDoor': {'z': 1500, 'theta': 30, 'diameter': 1300, 'color': '#FF4444'},
    'HVAC Door':  {'z': 2500, 'theta': 20, 'diameter': 400,  'color': '#44AAFF'},
    'RF Window':  {'z': 4000, 'theta': 40, 'diameter': 400,  'color': '#44FF44'},
    'Vent Hole 1': {'z': 300, 'theta': 15, 'diameter': 100,  'color': '#FFAA44'},
    'Vent Hole 2': {'z': 300, 'theta': 45, 'diameter': 100,  'color': '#FFAA44'},
}

# Ring frame z-positions
RING_FRAMES_P1 = [500, 2500, 3000, 3500, 4000, 4500]
RING_FRAMES_P2 = [500, 3000, 3500, 4500]


def get_radius_at_z(z):
    if z <= BARREL_H:
        return RADIUS
    dz = z - BARREL_H
    return RADIUS - OGIVE_R + np.sqrt(OGIVE_R**2 - dz**2)


def load_data(data_dir):
    nodes = pd.read_csv(os.path.join(data_dir, 'nodes.csv'))
    elems = pd.read_csv(os.path.join(data_dir, 'elements.csv'))
    nodes['r'] = np.sqrt(nodes['x']**2 + nodes['z']**2)
    nodes['theta_deg'] = np.degrees(np.arctan2(nodes['z'], nodes['x']))
    nodes['u_mag'] = np.sqrt(nodes['ux']**2 + nodes['uy']**2 + nodes['uz']**2)
    return nodes, elems


def build_surface_mesh(nodes, elems):
    """Build PyVista PolyData from nodes and S4R quad elements."""
    coords = nodes[['x', 'y', 'z']].values
    node_id_map = {}
    for idx, nid in enumerate(nodes['node_id'].values):
        node_id_map[nid] = idx

    faces = []
    valid_count = 0
    for _, row in elems.iterrows():
        try:
            n1 = node_id_map[int(row['n1'])]
            n2 = node_id_map[int(row['n2'])]
            n3 = node_id_map[int(row['n3'])]
            n4 = node_id_map[int(row['n4'])]
            faces.extend([4, n1, n2, n3, n4])
            valid_count += 1
        except KeyError:
            continue

    faces = np.array(faces, dtype=np.int64)
    mesh = pv.PolyData(coords, faces)
    print("  Mesh: %d points, %d cells" % (mesh.n_points, mesh.n_cells))
    return mesh, node_id_map


def identify_void_nodes(nodes, openings, tol_factor=1.2):
    """Identify nodes inside opening (void) regions."""
    void_mask = np.zeros(len(nodes), dtype=bool)
    for name, op in openings.items():
        r_half = op['diameter'] / 2.0
        z_c = op['z']
        theta_c = np.radians(op['theta'])
        for i, row in nodes.iterrows():
            r_at_z = get_radius_at_z(row['y'])
            d_theta = abs(np.arctan2(row['z'], row['x']) - theta_c)
            arc_dist = r_at_z * d_theta
            z_dist = abs(row['y'] - z_c)
            if arc_dist < r_half * tol_factor and z_dist < r_half * tol_factor:
                if row['smises'] < 0.01 and abs(row['temp']) < 0.01:
                    void_mask[i] = True
    return void_mask


def identify_void_nodes_fast(nodes, openings):
    """Fast identification: void nodes have smises~0 and temp~0."""
    return (nodes['smises'].abs() < 0.001) & (nodes['temp'].abs() < 0.5)


def add_fairing_wireframe(plotter, n_theta=7, n_z=30, alpha=0.15, color='gray'):
    """Add semi-transparent reference wireframe of the full fairing geometry."""
    theta_vals = np.linspace(0, np.radians(60), n_theta)
    z_vals = np.linspace(0, TOTAL_H, n_z)

    # Axial lines
    for th in theta_vals:
        pts = []
        for z in z_vals:
            r = get_radius_at_z(z) + CORE_T / 2
            pts.append([r * np.cos(th), z, r * np.sin(th)])
        line = pv.Spline(np.array(pts), n_points=200)
        plotter.add_mesh(line, color=color, opacity=alpha, line_width=0.5)

    # Circumferential lines
    for z in z_vals[::3]:
        r = get_radius_at_z(z) + CORE_T / 2
        th_range = np.linspace(0, np.radians(60), 60)
        pts = np.column_stack([r * np.cos(th_range), np.full_like(th_range, z), r * np.sin(th_range)])
        line = pv.Spline(pts, n_points=100)
        plotter.add_mesh(line, color=color, opacity=alpha, line_width=0.5)


def add_ring_frame_markers(plotter, z_positions, r_base=RADIUS, color='cyan', width=3):
    """Add ring frame location markers as arcs."""
    for z in z_positions:
        r = get_radius_at_z(z)
        th_range = np.linspace(0, np.radians(60), 80)
        pts = np.column_stack([r * np.cos(th_range), np.full_like(th_range, z), r * np.sin(th_range)])
        line = pv.Spline(pts, n_points=100)
        plotter.add_mesh(line, color=color, line_width=width, opacity=0.8)


def add_opening_markers(plotter, openings, offset=30):
    """Add opening boundary circles as colored rings."""
    for name, op in openings.items():
        z_c = op['z']
        theta_c = np.radians(op['theta'])
        r_half = op['diameter'] / 2.0
        r_at_z = get_radius_at_z(z_c) + CORE_T + offset

        n_pts = 60
        angles = np.linspace(0, 2 * np.pi, n_pts)
        pts = []
        for a in angles:
            dz = r_half * np.sin(a)
            d_arc = r_half * np.cos(a)
            d_theta = d_arc / r_at_z
            z_pt = z_c + dz
            th_pt = theta_c + d_theta
            r_pt = get_radius_at_z(z_pt) + CORE_T + offset
            pts.append([r_pt * np.cos(th_pt), z_pt, r_pt * np.sin(th_pt)])
        pts = np.array(pts)
        ring = pv.Spline(pts, n_points=120)
        plotter.add_mesh(ring, color=op['color'], line_width=4, opacity=0.9)


def setup_camera(plotter, view='iso'):
    """Set camera positions for different views.

    Fairing extents: x ∈ [0, 2638], y ∈ [0, 10400], z ∈ [0, 2638]
    Focal point at geometric center of the 1/6 sector.
    """
    # Center of fairing geometry (1/6 sector, theta ≈ 30°)
    r_mid = RADIUS * 0.85
    focal = (r_mid * np.cos(np.radians(30)),  # ~1940
             TOTAL_H * 0.45,                   # ~4680
             r_mid * np.sin(np.radians(30)))   # ~1105
    if view == 'iso':
        plotter.camera_position = [
            (14000, 8000, 14000),   # far enough to see entire 10.4m fairing
            focal,
            (0, 1, 0)
        ]
    elif view == 'front':
        plotter.camera_position = [
            (18000, TOTAL_H * 0.45, 0),
            focal,
            (0, 1, 0)
        ]
    elif view == 'side':
        plotter.camera_position = [
            (0, TOTAL_H * 0.45, 18000),
            focal,
            (0, 1, 0)
        ]
    elif view == 'top':
        plotter.camera_position = [
            (focal[0], 25000, focal[2]),
            focal,
            (1, 0, 0)
        ]
    elif view == 'opening_detail':
        # Focus on AccessDoor at z=1500, theta=30°
        r = get_radius_at_z(1500) + CORE_T
        cx = r * np.cos(np.radians(30))
        cz = r * np.sin(np.radians(30))
        plotter.camera_position = [
            (cx * 2.5, 1500, cz * 2.5),
            (cx, 1500, cz),
            (0, 1, 0)
        ]
    elif view == 'inner':
        # View from inside looking outward at theta=30°
        plotter.camera_position = [
            (500 * np.cos(np.radians(30)), TOTAL_H * 0.3, 500 * np.sin(np.radians(30))),
            (RADIUS * np.cos(np.radians(30)), TOTAL_H * 0.3, RADIUS * np.sin(np.radians(30))),
            (0, 1, 0)
        ]


def render_mesh_overview(mesh, nodes, phase_label, ring_frames, openings_dict, save_path):
    """Render mesh overview with wireframe, annotations."""
    pl = pv.Plotter(off_screen=True, window_size=[2400, 1800])
    pl.set_background('white', top='aliceblue')

    # Main mesh with edge visibility
    pl.add_mesh(mesh, color='lightskyblue', edge_color='steelblue',
                show_edges=True, opacity=0.85, line_width=0.3)

    add_ring_frame_markers(pl, ring_frames, color='#00CED1', width=4)

    if openings_dict:
        add_opening_markers(pl, openings_dict, offset=20)

    setup_camera(pl, 'iso')
    pl.add_text(
        'H3 Realistic Fairing: %s\nMesh Overview (%d nodes, %d elements)' %
        (phase_label, mesh.n_points, mesh.n_cells),
        position='upper_left', font_size=14, color='black', font='times'
    )

    # Legend entries
    legend_entries = [['Mesh surface', 'lightskyblue'], ['Ring frames', '#00CED1']]
    if openings_dict:
        for name, op in list(openings_dict.items())[:3]:
            legend_entries.append([name, op['color']])
    pl.add_legend(legend_entries, bcolor='white', face='circle', size=(0.18, 0.18))

    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def render_scalar_field(mesh, scalar_data, scalar_name, cmap, clim, phase_label,
                        save_path, view='iso', log_scale=False):
    """Render a scalar field on the mesh with colorbar."""
    pl = pv.Plotter(off_screen=True, window_size=[2400, 1800])
    pl.set_background('white', top='aliceblue')

    mesh_copy = mesh.copy()
    mesh_copy[scalar_name] = scalar_data

    if log_scale:
        safe_data = np.clip(scalar_data, 1e-6, None)
        mesh_copy[scalar_name] = np.log10(safe_data)
        if clim:
            clim = [np.log10(max(clim[0], 1e-6)), np.log10(clim[1])]

    pl.add_mesh(mesh_copy, scalars=scalar_name, cmap=cmap, clim=clim,
                show_edges=False, smooth_shading=True, opacity=1.0,
                scalar_bar_args={
                    'title': scalar_name,
                    'title_font_size': 16,
                    'label_font_size': 14,
                    'shadow': True,
                    'width': 0.4,
                    'height': 0.06,
                    'position_x': 0.3,
                    'position_y': 0.05,
                    'color': 'black',
                    'fmt': '%.1f',
                })

    setup_camera(pl, view)
    pl.add_text(
        'H3 Realistic Fairing: %s\n%s' % (phase_label, scalar_name),
        position='upper_left', font_size=14, color='black', font='times'
    )
    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def render_mesh_with_edges_detail(mesh, nodes, phase_label, openings_dict, save_path):
    """Render zoomed-in view of AccessDoor opening with mesh edges visible."""
    pl = pv.Plotter(off_screen=True, window_size=[2400, 1800])
    pl.set_background('white', top='aliceblue')

    smises = nodes['smises'].values
    mesh_copy = mesh.copy()
    mesh_copy['von Mises Stress (MPa)'] = smises

    vmax = min(np.percentile(smises[smises > 0.01], 98), 100)

    pl.add_mesh(mesh_copy, scalars='von Mises Stress (MPa)', cmap='hot',
                clim=[0, vmax], show_edges=True, edge_color='gray',
                line_width=0.3, opacity=1.0, smooth_shading=False,
                scalar_bar_args={
                    'title': 'von Mises (MPa)',
                    'title_font_size': 16,
                    'label_font_size': 14,
                    'shadow': True,
                    'width': 0.35,
                    'height': 0.06,
                    'position_x': 0.32,
                    'position_y': 0.05,
                    'color': 'black',
                    'fmt': '%.1f',
                })

    if openings_dict:
        add_opening_markers(pl, openings_dict, offset=5)

    setup_camera(pl, 'opening_detail')
    pl.add_text(
        'H3 Realistic Fairing: %s\nAccessDoor Detail — Mesh & Stress' % phase_label,
        position='upper_left', font_size=14, color='black', font='times'
    )
    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def render_multi_view(mesh, scalar_data, scalar_name, cmap, clim, phase_label, save_path):
    """Render 4-panel multi-view: iso, front, side, top."""
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=[3200, 2400])
    views = [('iso', 'Isometric'), ('front', 'Front View'),
             ('side', 'Side View'), ('top', 'Top View')]

    for i, (view, title) in enumerate(views):
        row, col = divmod(i, 2)
        pl.subplot(row, col)
        pl.set_background('white', top='aliceblue')
        mesh_copy = mesh.copy()
        mesh_copy[scalar_name] = scalar_data
        pl.add_mesh(mesh_copy, scalars=scalar_name, cmap=cmap, clim=clim,
                    show_edges=False, smooth_shading=True,
                    scalar_bar_args={
                        'title': scalar_name if i == 0 else '',
                        'title_font_size': 12,
                        'label_font_size': 10,
                        'width': 0.3,
                        'height': 0.05,
                        'position_x': 0.35,
                        'position_y': 0.05,
                        'color': 'black',
                        'fmt': '%.1f',
                    })
        setup_camera(pl, view)
        pl.add_text(title, position='upper_left', font_size=12, color='black')

    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def render_phase_comparison(mesh1, data1, mesh2, data2, scalar_name, cmap, clim,
                            label1, label2, save_path):
    """Side-by-side Phase 1 vs Phase 2 comparison."""
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[3600, 1800])

    for idx, (mesh, data, label) in enumerate([(mesh1, data1, label1), (mesh2, data2, label2)]):
        pl.subplot(0, idx)
        pl.set_background('white', top='aliceblue')
        mc = mesh.copy()
        mc[scalar_name] = data
        pl.add_mesh(mc, scalars=scalar_name, cmap=cmap, clim=clim,
                    show_edges=False, smooth_shading=True,
                    scalar_bar_args={
                        'title': scalar_name,
                        'title_font_size': 14,
                        'label_font_size': 12,
                        'width': 0.35,
                        'height': 0.06,
                        'position_x': 0.32,
                        'position_y': 0.05,
                        'color': 'black',
                        'fmt': '%.1f',
                    })
        setup_camera(pl, 'iso')
        pl.add_text(label, position='upper_left', font_size=14, color='black', font='times')

    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def render_deformed_shape(mesh, nodes, scale_factor, phase_label, save_path):
    """Render deformed mesh overlaid on undeformed (ghost)."""
    pl = pv.Plotter(off_screen=True, window_size=[2400, 1800])
    pl.set_background('white', top='aliceblue')

    # Undeformed (ghost)
    pl.add_mesh(mesh, color='lightgray', opacity=0.2, show_edges=True,
                edge_color='silver', line_width=0.2)

    # Deformed
    disp = nodes[['ux', 'uy', 'uz']].values * scale_factor
    deformed_pts = mesh.points + disp
    deformed_mesh = mesh.copy()
    deformed_mesh.points = deformed_pts
    deformed_mesh['Displacement (mm)'] = nodes['u_mag'].values

    pl.add_mesh(deformed_mesh, scalars='Displacement (mm)', cmap='turbo',
                show_edges=False, smooth_shading=True,
                scalar_bar_args={
                    'title': 'Displacement (mm), Scale: %dx' % scale_factor,
                    'title_font_size': 14,
                    'label_font_size': 12,
                    'width': 0.4,
                    'height': 0.06,
                    'position_x': 0.3,
                    'position_y': 0.05,
                    'color': 'black',
                    'fmt': '%.1f',
                })

    setup_camera(pl, 'iso')
    pl.add_text(
        'H3 Realistic Fairing: %s\nDeformed Shape (x%d)' % (phase_label, scale_factor),
        position='upper_left', font_size=14, color='black', font='times'
    )
    legend_entries = [['Undeformed', 'lightgray'], ['Deformed (colored)', 'orange']]
    pl.add_legend(legend_entries, bcolor='white', face='circle', size=(0.15, 0.1))
    pl.screenshot(save_path, transparent_background=False, scale=2)
    pl.close()
    print("  Saved: %s" % os.path.basename(save_path))


def main():
    print("=" * 60)
    print("  H3 Realistic Fairing — 3D Visualization")
    print("=" * 60)

    # --- Load data ---
    print("\n[Phase 1] Loading data...")
    nodes1, elems1 = load_data(PHASE1_DIR)
    print("  Nodes: %d, Elements: %d" % (len(nodes1), len(elems1)))

    print("\n[Phase 2] Loading data...")
    nodes2, elems2 = load_data(PHASE2_DIR)
    print("  Nodes: %d, Elements: %d" % (len(nodes2), len(elems2)))

    # --- Build meshes ---
    print("\n[Phase 1] Building mesh...")
    mesh1, nmap1 = build_surface_mesh(nodes1, elems1)

    print("\n[Phase 2] Building mesh...")
    mesh2, nmap2 = build_surface_mesh(nodes2, elems2)

    # --- Compute scalar limits for consistent coloring ---
    vmax_s = max(nodes1['smises'].quantile(0.98), nodes2['smises'].quantile(0.98))
    vmax_d = max(nodes1['u_mag'].max(), nodes2['u_mag'].max())

    # ============================================================
    # 1. Mesh overview (Phase 1 & Phase 2)
    # ============================================================
    print("\n--- Mesh Overview ---")
    render_mesh_overview(mesh1, nodes1, 'Phase 1 (AccessDoor + 6 Frames)',
                         RING_FRAMES_P1, {'AccessDoor': OPENINGS['AccessDoor']},
                         os.path.join(FIG_DIR, '01_mesh_overview_p1.png'))

    render_mesh_overview(mesh2, nodes2, 'Phase 2 (5 Openings + 4 Frames)',
                         RING_FRAMES_P2, OPENINGS,
                         os.path.join(FIG_DIR, '02_mesh_overview_p2.png'))

    # ============================================================
    # 2. von Mises stress (iso view)
    # ============================================================
    print("\n--- von Mises Stress ---")
    render_scalar_field(mesh1, nodes1['smises'].values,
                        'von Mises Stress (MPa)', 'hot', [0, vmax_s],
                        'Phase 1', os.path.join(FIG_DIR, '03_stress_p1.png'))

    render_scalar_field(mesh2, nodes2['smises'].values,
                        'von Mises Stress (MPa)', 'hot', [0, vmax_s],
                        'Phase 2', os.path.join(FIG_DIR, '04_stress_p2.png'))

    # ============================================================
    # 3. Displacement magnitude
    # ============================================================
    print("\n--- Displacement ---")
    render_scalar_field(mesh1, nodes1['u_mag'].values,
                        'Displacement (mm)', 'turbo', [0, vmax_d],
                        'Phase 1', os.path.join(FIG_DIR, '05_disp_p1.png'))

    render_scalar_field(mesh2, nodes2['u_mag'].values,
                        'Displacement (mm)', 'turbo', [0, vmax_d],
                        'Phase 2', os.path.join(FIG_DIR, '06_disp_p2.png'))

    # ============================================================
    # 4. Temperature
    # ============================================================
    print("\n--- Temperature ---")
    render_scalar_field(mesh1, nodes1['temp'].values,
                        'Temperature (C)', 'coolwarm', [0, 120],
                        'Phase 1', os.path.join(FIG_DIR, '07_temp_p1.png'))

    render_scalar_field(mesh2, nodes2['temp'].values,
                        'Temperature (C)', 'coolwarm', [0, 120],
                        'Phase 2', os.path.join(FIG_DIR, '08_temp_p2.png'))

    # ============================================================
    # 5. Mesh detail with edges (AccessDoor zoom)
    # ============================================================
    print("\n--- Mesh Detail ---")
    render_mesh_with_edges_detail(mesh1, nodes1, 'Phase 1',
                                  {'AccessDoor': OPENINGS['AccessDoor']},
                                  os.path.join(FIG_DIR, '09_detail_p1.png'))

    render_mesh_with_edges_detail(mesh2, nodes2, 'Phase 2', OPENINGS,
                                  os.path.join(FIG_DIR, '10_detail_p2.png'))

    # ============================================================
    # 6. Multi-view panels
    # ============================================================
    print("\n--- Multi-View ---")
    render_multi_view(mesh2, nodes2['smises'].values,
                      'von Mises Stress (MPa)', 'hot', [0, vmax_s],
                      'Phase 2', os.path.join(FIG_DIR, '11_multiview_stress_p2.png'))

    render_multi_view(mesh2, nodes2['u_mag'].values,
                      'Displacement (mm)', 'turbo', [0, vmax_d],
                      'Phase 2', os.path.join(FIG_DIR, '12_multiview_disp_p2.png'))

    # ============================================================
    # 7. Phase 1 vs Phase 2 side-by-side
    # ============================================================
    print("\n--- Phase Comparison ---")
    render_phase_comparison(mesh1, nodes1['smises'].values,
                            mesh2, nodes2['smises'].values,
                            'von Mises Stress (MPa)', 'hot', [0, vmax_s],
                            'Phase 1 (AccessDoor + 6 Frames)',
                            'Phase 2 (5 Openings + 4 Frames)',
                            os.path.join(FIG_DIR, '13_comparison_stress.png'))

    render_phase_comparison(mesh1, nodes1['u_mag'].values,
                            mesh2, nodes2['u_mag'].values,
                            'Displacement (mm)', 'turbo', [0, vmax_d],
                            'Phase 1', 'Phase 2',
                            os.path.join(FIG_DIR, '14_comparison_disp.png'))

    # ============================================================
    # 8. Deformed shape (exaggerated)
    # ============================================================
    print("\n--- Deformed Shape ---")
    render_deformed_shape(mesh1, nodes1, 50, 'Phase 1',
                          os.path.join(FIG_DIR, '15_deformed_p1.png'))
    render_deformed_shape(mesh2, nodes2, 50, 'Phase 2',
                          os.path.join(FIG_DIR, '16_deformed_p2.png'))

    print("\n" + "=" * 60)
    print("  All 16 figures saved to: %s" % FIG_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
