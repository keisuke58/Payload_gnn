#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 フェアリング 360° シネマティック 3D 動画 (v2)

v1 からの改善点:
  - 変形メッシュ表示（応力による変形を誇張）
  - 欠陥領域ハイライト（赤リング + 半透明球）
  - リングフレームのワイヤーフレーム表示
  - 映画的カメラワーク（アプローチ → 周回 → 欠陥ズーム → 引き）
  - スムーズイージング（ease-in-out）
  - 3灯ライティング + アンビエント
  - 白背景 + 応力コンターライン（等高線）
"""
import os
import argparse
import numpy as np
import pandas as pd

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
import pyvista as pv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# H3 fairing geometry
RADIUS = 2600.0
BARREL_H = 5000.0
OGIVE_H = 5400.0
TOTAL_H = BARREL_H + OGIVE_H
N_SECTORS = 6
SECTOR_ANGLE = 2 * np.pi / N_SECTORS
OGIVE_R = (RADIUS**2 + OGIVE_H**2) / (2 * RADIUS)

# Ring frame z-positions (mm)
RING_FRAMES = [500, 2500, 3000, 3500, 4000, 4500]


def get_radius_at_z(z):
    if np.isscalar(z):
        if z <= BARREL_H:
            return RADIUS
        dz = z - BARREL_H
        return RADIUS - OGIVE_R + np.sqrt(max(OGIVE_R**2 - dz**2, 0))
    result = np.full_like(z, RADIUS, dtype=float)
    mask = z > BARREL_H
    dz = z[mask] - BARREL_H
    result[mask] = RADIUS - OGIVE_R + np.sqrt(np.maximum(OGIVE_R**2 - dz**2, 0))
    return result


def ease_in_out(t):
    """Smooth ease-in-out: 0→1 with acceleration/deceleration"""
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    """Linear interpolation"""
    return a + (b - a) * t


def slerp_camera(pos_a, pos_b, focus_a, focus_b, t):
    """Smooth camera interpolation"""
    s = ease_in_out(t)
    pos = tuple(lerp(np.array(pos_a), np.array(pos_b), s))
    foc = tuple(lerp(np.array(focus_a), np.array(focus_b), s))
    return pos, foc


def load_sample(sample_dir):
    nodes = pd.read_csv(os.path.join(sample_dir, 'nodes.csv'))
    elems = pd.read_csv(os.path.join(sample_dir, 'elements.csv'))
    meta = {}
    meta_path = os.path.join(sample_dir, 'metadata.csv')
    if os.path.exists(meta_path):
        mdf = pd.read_csv(meta_path)
        for _, row in mdf.iterrows():
            meta[row['key']] = row['value']
    return nodes, elems, meta


def harmonize_boundaries(nodes, scalar_dict, disp, blend_deg=5.0):
    """θ=0° と θ=60° 境界の値を平滑化して接合部の不連続を解消する。

    θ=0° 側のノードと θ=60° 側のノードを (y, r) でマッチングし、
    境界付近のスカラー値をブレンドする。
    """
    from scipy.spatial import cKDTree

    x, y, z = nodes['x'].values, nodes['y'].values, nodes['z'].values
    r = np.sqrt(x**2 + z**2)
    theta = np.degrees(np.arctan2(z, x))

    lo_mask = theta < blend_deg
    hi_mask = theta > (60 - blend_deg)
    lo_idx = np.where(lo_mask)[0]
    hi_idx = np.where(hi_mask)[0]

    if len(lo_idx) == 0 or len(hi_idx) == 0:
        return scalar_dict, disp

    # (y, r) 空間で θ≈0° ↔ θ≈60° のマッチング
    lo_yr = np.column_stack([y[lo_idx], r[lo_idx]])
    hi_yr = np.column_stack([y[hi_idx], r[hi_idx]])
    tree_hi = cKDTree(hi_yr)
    tree_lo = cKDTree(lo_yr)

    dists_lo2hi, match_lo2hi = tree_hi.query(lo_yr)
    dists_hi2lo, match_hi2lo = tree_lo.query(hi_yr)

    max_match_dist = 60.0  # mm

    # スカラーのブレンド
    for k, vals in scalar_dict.items():
        if k == 'defect_label':
            continue
        blended = vals.copy()

        # θ≈0° ノード: θ=0 で avg、θ=blend_deg で元の値
        for i, hi_j in enumerate(match_lo2hi):
            if dists_lo2hi[i] > max_match_dist:
                continue
            li = lo_idx[i]
            hj = hi_idx[hi_j]
            avg = (vals[li] + vals[hj]) / 2
            t = theta[li] / blend_deg  # 0→1
            blended[li] = avg + (vals[li] - avg) * t

        # θ≈60° ノード: θ=60 で avg、θ=60-blend_deg で元の値
        for i, lo_j in enumerate(match_hi2lo):
            if dists_hi2lo[i] > max_match_dist:
                continue
            hj = hi_idx[i]
            li = lo_idx[lo_j]
            avg = (vals[li] + vals[hj]) / 2
            t = (60 - theta[hj]) / blend_deg  # 0→1
            blended[hj] = avg + (vals[hj] - avg) * t

        scalar_dict[k] = blended

    # 変位ベクトルも同様にブレンド
    disp_blended = disp.copy()
    for dim in range(3):
        d = disp[:, dim].copy()
        for i, hi_j in enumerate(match_lo2hi):
            if dists_lo2hi[i] > max_match_dist:
                continue
            li = lo_idx[i]
            hj = hi_idx[hi_j]
            avg = (d[li] + d[hj]) / 2
            t = theta[li] / blend_deg
            d[li] = avg + (d[li] - avg) * t
        for i, lo_j in enumerate(match_hi2lo):
            if dists_hi2lo[i] > max_match_dist:
                continue
            hj = hi_idx[i]
            li = lo_idx[lo_j]
            avg = (d[li] + d[hj]) / 2
            t = (60 - theta[hj]) / blend_deg
            d[hj] = avg + (d[hj] - avg) * t
        disp_blended[:, dim] = d

    n_blended = np.sum(dists_lo2hi < max_match_dist) + np.sum(dists_hi2lo < max_match_dist)
    print(f"  Boundary blending: {n_blended} nodes blended (blend_deg={blend_deg}°)")
    return scalar_dict, disp_blended


def replicate_360(nodes, elems, blend_boundaries=True):
    """60° sector → 360° full model (境界ブレンディング付き)"""
    coords_base = nodes[['x', 'y', 'z']].values
    disp_base = nodes[['ux', 'uy', 'uz']].values
    n_nodes = len(nodes)

    node_ids = nodes['node_id'].values
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    elem_indices = []
    for _, row in elems.iterrows():
        try:
            idx = [id_to_idx[int(row[c])] for c in ('n1', 'n2', 'n3', 'n4')]
            elem_indices.append(idx)
        except KeyError:
            continue
    elem_indices = np.array(elem_indices, dtype=np.int64)
    n_elems = len(elem_indices)

    scalar_keys = ['smises', 'temp', 'u_mag', 'defect_label']
    scalars_base = {k: nodes[k].values.copy() for k in scalar_keys if k in nodes.columns}

    # 境界ブレンディング（複製前に平滑化）
    if blend_boundaries:
        scalars_base, disp_base = harmonize_boundaries(
            nodes, scalars_base, disp_base, blend_deg=5.0)

    all_coords = []
    all_disp = []
    all_faces = []
    all_scalars = {k: [] for k in scalars_base}

    for i in range(N_SECTORS):
        angle = i * SECTOR_ANGLE
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, 0, -sin_a], [0, 1, 0], [sin_a, 0, cos_a]])

        all_coords.append(coords_base @ rot.T)
        all_disp.append(disp_base @ rot.T)

        offset = i * n_nodes
        shifted = elem_indices + offset
        for quad in shifted:
            all_faces.extend([4, quad[0], quad[1], quad[2], quad[3]])

        for k in all_scalars:
            all_scalars[k].append(scalars_base[k])

    all_coords = np.vstack(all_coords)
    all_disp = np.vstack(all_disp)
    all_faces = np.array(all_faces, dtype=np.int64)
    for k in all_scalars:
        all_scalars[k] = np.concatenate(all_scalars[k])

    print(f"  360° model: {all_coords.shape[0]:,} nodes, {n_elems * N_SECTORS:,} elements")
    return all_coords, all_disp, all_faces, all_scalars


def build_mesh(coords, faces, scalars):
    mesh = pv.PolyData(coords, faces)
    for k, v in scalars.items():
        mesh[k] = v
    mesh = mesh.compute_normals(auto_orient_normals=True)
    return mesh


def add_ring_frames(plotter, full_circle=True):
    """リングフレームをラインで追加"""
    theta_max = 2 * np.pi if full_circle else np.radians(60)
    n_pts = 360 if full_circle else 60
    for z in RING_FRAMES:
        r = get_radius_at_z(z) + 20
        th = np.linspace(0, theta_max, n_pts)
        pts = np.column_stack([r * np.cos(th), np.full(n_pts, z), r * np.sin(th)])
        line = pv.Spline(pts, n_points=n_pts * 2)
        plotter.add_mesh(line, color='#006699', line_width=2, opacity=0.4)


def add_defect_marker(plotter, defect_pos, defect_radius):
    """欠陥位置に赤いリング + 半透明球を追加"""
    sphere = pv.Sphere(radius=defect_radius * 2.5, center=defect_pos)
    plotter.add_mesh(sphere, color='red', opacity=0.10, smooth_shading=True)

    for r_mult, w, op in [(2.0, 3, 0.9), (2.8, 2, 0.6), (3.5, 1, 0.3)]:
        ring_r = defect_radius * r_mult
        th = np.linspace(0, 2 * np.pi, 120)
        pts = np.column_stack([
            defect_pos[0] + ring_r * np.cos(th),
            np.full(120, defect_pos[1]),
            defect_pos[2] + ring_r * np.sin(th),
        ])
        ring = pv.Spline(pts, n_points=200)
        plotter.add_mesh(ring, color='red', line_width=w, opacity=op)


def add_stress_contours(plotter, mesh, scalar_name='smises', n_contours=12):
    """応力コンターライン（等高線）をメッシュに重ねる"""
    contours = mesh.contour(isosurfaces=n_contours, scalars=scalar_name)
    if contours.n_points > 0:
        plotter.add_mesh(contours, color='black', line_width=1.2, opacity=0.45)
        print(f"  Contours: {n_contours} iso-lines, {contours.n_points:,} pts")


def compute_defect_3d_pos(meta):
    """metadata から欠陥の 3D 座標を計算"""
    theta_deg = float(meta.get('theta_deg', 30))
    z_center = float(meta.get('z_center', 2000))
    radius = float(meta.get('radius', 100))
    r = get_radius_at_z(z_center)
    theta = np.radians(theta_deg)
    return (r * np.cos(theta), z_center, r * np.sin(theta)), radius


def render_cinematic(mesh, coords, disp, scalars, meta, output_path,
                     n_frames=540, fps=30, resolution=(1920, 1080),
                     deform_scale=50, show_contour_lines=False):
    """
    映画的カメラワークで応力分布を描画。

    Phase 構成 (540 frames @ 30fps = 18秒):
      Phase A (0-89):    遠方からアプローチ (3.0s)
      Phase B (90-359):  周回 (9.0s) — メインの 360° orbit
      Phase C (360-449): 欠陥領域にズーム (3.0s)
      Phase D (450-539): 引きながら見上げ (3.0s)
    """
    # 変形メッシュ
    deformed_coords = coords + disp * deform_scale
    mesh_deformed = mesh.copy()
    mesh_deformed.points = deformed_coords

    # 欠陥位置
    defect_pos, defect_r = compute_defect_3d_pos(meta)

    # スカラー範囲 — きりの良い目盛りにする
    smises = scalars['smises']
    pos_data = smises[smises > 0.01]
    raw_max = np.percentile(pos_data, 97)
    # 目盛りがきれいになるよう 5 MPa 刻みに丸める
    vmax = np.ceil(raw_max / 5) * 5
    vmin = 0.0
    n_ticks = int(vmax / 5) + 1

    print(f"  Stress range: [0, {vmax:.0f}] MPa (raw p97={raw_max:.1f})")
    print(f"  Colorbar ticks: {n_ticks} (0, 5, 10, ... {vmax:.0f})")
    print(f"  Deformation scale: {deform_scale}x")
    print(f"  Defect at: theta={meta.get('theta_deg','?')}°, "
          f"z={meta.get('z_center','?')}mm, r={meta.get('radius','?')}mm")

    # --- Plotter setup ---
    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('white', top='#f0f4f8')

    # メインメッシュ（変形 + 応力）
    pl.add_mesh(
        mesh_deformed, scalars='smises', cmap='jet',
        clim=[vmin, vmax], smooth_shading=True, show_edges=False,
        n_colors=256,
        scalar_bar_args={
            'title': 'von Mises Stress (MPa)',
            'title_font_size': 18, 'label_font_size': 14,
            'shadow': True,
            'n_labels': n_ticks,
            'width': 0.45, 'height': 0.07,
            'position_x': 0.27, 'position_y': 0.03,
            'color': 'black',
            'fmt': '%.0f',
            'bold': True,
        },
    )

    # ゴースト（非変形）メッシュ
    pl.add_mesh(mesh, color='#cccccc', opacity=0.05, smooth_shading=True)

    # コンターライン（オプション）
    if show_contour_lines:
        add_stress_contours(pl, mesh_deformed, 'smises', n_contours=12)

    # リングフレーム
    add_ring_frames(pl, full_circle=True)

    # 欠陥マーカー
    add_defect_marker(pl, defect_pos, defect_r)

    # ライティング（白背景向け — 明るめ）
    pl.add_light(pv.Light(position=(25000, 20000, 25000), intensity=0.7,
                          color='white'))
    pl.add_light(pv.Light(position=(-20000, 15000, -10000), intensity=0.35,
                          color='white'))
    pl.add_light(pv.Light(position=(0, -5000, 15000), intensity=0.2,
                          color='white'))

    # テキスト
    pl.add_text(
        'H3 Rocket Fairing — Launch Stress Analysis\n'
        f'von Mises Stress | Deformation x{deform_scale}',
        position='upper_left', font_size=13, color='black', font='times',
    )

    # --- Phase 定義 ---
    phase_a = 90    # approach
    phase_b = 270   # orbit
    phase_c = 90    # zoom to defect
    phase_d = 90    # pull back
    total = phase_a + phase_b + phase_c + phase_d
    assert total == n_frames, f"Total frames mismatch: {total} != {n_frames}"

    center = np.array([0, TOTAL_H * 0.42, 0])
    cam_far = 40000
    cam_orbit = 22000
    cam_close = 8000

    print(f"  Rendering {n_frames} frames ({n_frames/fps:.1f}s) → {output_path}")
    pl.open_movie(output_path, framerate=fps, quality=8)

    for frame in range(n_frames):
        if frame < phase_a:
            # Phase A: 遠方からアプローチ
            t = frame / phase_a
            s = ease_in_out(t)
            dist = lerp(cam_far, cam_orbit, s)
            azimuth = np.radians(30)  # 固定角度から接近
            elevation = lerp(np.radians(35), np.radians(22), s)
            focus = tuple(lerp(np.array([0, TOTAL_H * 0.5, 0]), center, s))

        elif frame < phase_a + phase_b:
            # Phase B: 360° 周回
            t = (frame - phase_a) / phase_b
            dist = cam_orbit
            azimuth = 2 * np.pi * t + np.radians(30)
            elevation = np.radians(22) + np.radians(6) * np.sin(2 * np.pi * t)
            focus = tuple(center)

        elif frame < phase_a + phase_b + phase_c:
            # Phase C: 欠陥にズーム
            t = (frame - phase_a - phase_b) / phase_c
            s = ease_in_out(t)

            # orbit 最終位置
            orbit_end_az = 2 * np.pi + np.radians(30)
            orbit_end_el = np.radians(22)
            orbit_end_pos = np.array([
                cam_orbit * np.cos(orbit_end_el) * np.cos(orbit_end_az),
                center[1] + cam_orbit * np.sin(orbit_end_el),
                cam_orbit * np.cos(orbit_end_el) * np.sin(orbit_end_az),
            ])

            # 欠陥を見るカメラ位置
            dp = np.array(defect_pos)
            defect_dir = dp.copy()
            defect_dir[1] = 0
            defect_dir = defect_dir / (np.linalg.norm(defect_dir) + 1e-8)
            close_pos = dp + defect_dir * cam_close + np.array([0, 1500, 0])

            pos = tuple(lerp(orbit_end_pos, close_pos, s))
            focus = tuple(lerp(np.array(center), dp, s))

            pl.camera_position = [pos, focus, (0, 1, 0)]
            pl.write_frame()
            if (frame + 1) % 90 == 0:
                print(f"    frame {frame + 1}/{n_frames}")
            continue

        else:
            # Phase D: 引きながら見上げ
            t = (frame - phase_a - phase_b - phase_c) / phase_d
            s = ease_in_out(t)

            dp = np.array(defect_pos)
            defect_dir = dp.copy()
            defect_dir[1] = 0
            defect_dir = defect_dir / (np.linalg.norm(defect_dir) + 1e-8)
            close_pos = dp + defect_dir * cam_close + np.array([0, 1500, 0])

            far_pos = np.array([
                cam_orbit * 0.8 * np.cos(np.radians(15)),
                center[1] + cam_orbit * 0.6,
                cam_orbit * 0.8 * np.sin(np.radians(15)),
            ])

            pos = tuple(lerp(close_pos, far_pos, s))
            focus = tuple(lerp(dp, np.array(center), s))

            pl.camera_position = [pos, focus, (0, 1, 0)]
            pl.write_frame()
            if (frame + 1) % 90 == 0:
                print(f"    frame {frame + 1}/{n_frames}")
            continue

        cam_x = dist * np.cos(elevation) * np.cos(azimuth)
        cam_z = dist * np.cos(elevation) * np.sin(azimuth)
        cam_y = focus[1] + dist * np.sin(elevation)

        pl.camera_position = [(cam_x, cam_y, cam_z), focus, (0, 1, 0)]
        pl.write_frame()

        if (frame + 1) % 90 == 0:
            print(f"    frame {frame + 1}/{n_frames}")

    pl.close()
    print(f"  Done: {output_path}")


def render_temperature_cinematic(mesh, coords, scalars, meta, output_path,
                                 n_frames=360, fps=30, resolution=(1920, 1080)):
    """温度分布の周回動画（変形なし、白背景）"""
    temp = scalars['temp']
    # きりの良い温度範囲 (10°C 刻み)
    raw_min, raw_max = np.percentile(temp, 1), np.percentile(temp, 99)
    vmin = np.floor(raw_min / 10) * 10
    vmax = np.ceil(raw_max / 10) * 10
    n_ticks = int((vmax - vmin) / 20) + 1

    print(f"  Temp range: [{vmin:.0f}, {vmax:.0f}] °C (ticks every 20°C)")

    defect_pos, defect_r = compute_defect_3d_pos(meta)

    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('white', top='#f0f4f8')

    pl.add_mesh(
        mesh, scalars='temp', cmap='coolwarm',
        clim=[vmin, vmax], smooth_shading=True, show_edges=False,
        n_colors=256,
        scalar_bar_args={
            'title': 'Temperature (°C)',
            'title_font_size': 18, 'label_font_size': 14,
            'shadow': True,
            'n_labels': n_ticks,
            'width': 0.45, 'height': 0.07,
            'position_x': 0.27, 'position_y': 0.03,
            'color': 'black',
            'fmt': '%.0f',
            'bold': True,
        },
    )

    add_ring_frames(pl, full_circle=True)
    add_defect_marker(pl, defect_pos, defect_r)

    pl.add_light(pv.Light(position=(25000, 20000, 25000), intensity=0.7))
    pl.add_light(pv.Light(position=(-20000, 15000, -10000), intensity=0.35,
                          color='white'))
    pl.add_light(pv.Light(position=(0, -5000, 15000), intensity=0.2,
                          color='white'))

    pl.add_text(
        'H3 Rocket Fairing — Aerodynamic Heating\n'
        f'Temperature Distribution ({vmin:.0f}–{vmax:.0f}°C)',
        position='upper_left', font_size=13, color='black', font='times',
    )

    center = np.array([0, TOTAL_H * 0.42, 0])
    cam_dist = 22000

    print(f"  Rendering {n_frames} frames ({n_frames/fps:.1f}s) → {output_path}")
    pl.open_movie(output_path, framerate=fps, quality=8)

    for frame in range(n_frames):
        t = frame / n_frames
        azimuth = 2 * np.pi * t
        elevation = np.radians(20) + np.radians(8) * np.sin(2 * np.pi * t * 2)

        cam_x = cam_dist * np.cos(elevation) * np.cos(azimuth)
        cam_z = cam_dist * np.cos(elevation) * np.sin(azimuth)
        cam_y = center[1] + cam_dist * np.sin(elevation)

        pl.camera_position = [(cam_x, cam_y, cam_z), tuple(center), (0, 1, 0)]
        pl.write_frame()

        if (frame + 1) % 90 == 0:
            print(f"    frame {frame + 1}/{n_frames}")

    pl.close()
    print(f"  Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='H3 Fairing 360° cinematic video v2')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--dataset', default='dataset_realistic_25mm_100')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--deform-scale', type=int, default=50)
    parser.add_argument('--stress-only', action='store_true')
    parser.add_argument('--temp-only', action='store_true')
    parser.add_argument('--contour-lines', action='store_true',
                        help='応力コンターライン（等高線）を重ねる')
    args = parser.parse_args()

    sample_dir = os.path.join(
        PROJECT_ROOT, args.dataset, f'sample_{args.sample:04d}')
    out_dir = os.path.join(PROJECT_ROOT, 'figures', 'fairing_360_video')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  H3 Fairing — Cinematic 360° Video (v2)")
    print("=" * 60)

    print(f"\n[1] Loading sample {args.sample:04d} ...")
    nodes, elems, meta = load_sample(sample_dir)
    print(f"  Nodes: {len(nodes):,}, Elements: {len(elems):,}")
    print(f"  Defect: {meta.get('defect_type', '?')} at "
          f"theta={meta.get('theta_deg', '?')}°, z={meta.get('z_center', '?')}mm")

    print("\n[2] Building 360° full model ...")
    coords, disp, faces, scalars = replicate_360(nodes, elems)
    mesh = build_mesh(coords, faces, scalars)

    res = (args.width, args.height)

    if not args.temp_only:
        print("\n[3] Rendering stress cinematic ...")
        out_stress = os.path.join(
            out_dir, f'h3_fairing_cinematic_stress_sample{args.sample:04d}.mp4')
        render_cinematic(
            mesh, coords, disp, scalars, meta, out_stress,
            n_frames=540, fps=args.fps, resolution=res,
            deform_scale=args.deform_scale,
            show_contour_lines=args.contour_lines,
        )

    if not args.stress_only:
        print("\n[4] Rendering temperature orbit ...")
        out_temp = os.path.join(
            out_dir, f'h3_fairing_cinematic_temp_sample{args.sample:04d}.mp4')
        render_temperature_cinematic(
            mesh, coords, scalars, meta, out_temp,
            n_frames=360, fps=args.fps, resolution=res,
        )

    print("\n" + "=" * 60)
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
