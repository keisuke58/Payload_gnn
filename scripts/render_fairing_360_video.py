#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 フェアリング 360° 3D 回転アニメーション動画生成

1/6 セクター（60°）を 6 回複製して 360° フルモデルを構築し、
打ち上げ時の応力分布・温度分布を PyVista でレンダリング。
カメラが周回する MP4 動画を出力する。
"""
import os
import sys
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
SECTOR_ANGLE = 2 * np.pi / N_SECTORS  # 60°


def get_radius_at_z(z):
    """フェアリング高さ z での半径を返す（バレル + オジーブ）"""
    ogive_R = (RADIUS**2 + OGIVE_H**2) / (2 * RADIUS)
    if np.isscalar(z):
        if z <= BARREL_H:
            return RADIUS
        dz = z - BARREL_H
        return RADIUS - ogive_R + np.sqrt(max(ogive_R**2 - dz**2, 0))
    result = np.full_like(z, RADIUS, dtype=float)
    mask = z > BARREL_H
    dz = z[mask] - BARREL_H
    result[mask] = RADIUS - ogive_R + np.sqrt(np.maximum(ogive_R**2 - dz**2, 0))
    return result


def load_sample(sample_dir):
    """ノードと要素データを読み込む"""
    nodes = pd.read_csv(os.path.join(sample_dir, 'nodes.csv'))
    elems = pd.read_csv(os.path.join(sample_dir, 'elements.csv'))
    return nodes, elems


def replicate_360(nodes, elems):
    """60° セクターを 6 回回転させて 360° フルモデルを構築する。

    Returns:
        all_coords: (N*6, 3) 座標
        all_faces: PyVista faces 配列
        all_scalars: dict of scalar arrays
    """
    coords_base = nodes[['x', 'y', 'z']].values
    n_nodes = len(nodes)

    # ノード ID → 連番インデックス
    node_ids = nodes['node_id'].values
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # 要素の連番インデックス化
    elem_indices = []
    for _, row in elems.iterrows():
        try:
            idx = [id_to_idx[int(row[c])] for c in ('n1', 'n2', 'n3', 'n4')]
            elem_indices.append(idx)
        except KeyError:
            continue
    elem_indices = np.array(elem_indices, dtype=np.int64)
    n_elems = len(elem_indices)

    # スカラー量
    scalar_keys = ['smises', 'temp', 'u_mag', 'defect_label']
    scalars_base = {k: nodes[k].values for k in scalar_keys if k in nodes.columns}

    all_coords = []
    all_faces = []
    all_scalars = {k: [] for k in scalars_base}

    for i in range(N_SECTORS):
        angle = i * SECTOR_ANGLE
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # y 軸周りの回転: x' = x cos - z sin, z' = x sin + z cos
        rotated = np.column_stack([
            coords_base[:, 0] * cos_a - coords_base[:, 2] * sin_a,
            coords_base[:, 1],
            coords_base[:, 0] * sin_a + coords_base[:, 2] * cos_a,
        ])
        all_coords.append(rotated)

        # 要素インデックスをオフセット
        offset = i * n_nodes
        shifted = elem_indices + offset
        for quad in shifted:
            all_faces.extend([4, quad[0], quad[1], quad[2], quad[3]])

        for k in all_scalars:
            all_scalars[k].append(scalars_base[k])

    all_coords = np.vstack(all_coords)
    all_faces = np.array(all_faces, dtype=np.int64)
    for k in all_scalars:
        all_scalars[k] = np.concatenate(all_scalars[k])

    print(f"  360° model: {all_coords.shape[0]:,} nodes, {n_elems * N_SECTORS:,} elements")
    return all_coords, all_faces, all_scalars


def build_full_mesh(coords, faces, scalars):
    """PyVista PolyData を構築"""
    mesh = pv.PolyData(coords, faces)
    for k, v in scalars.items():
        mesh[k] = v
    mesh = mesh.compute_normals(auto_orient_normals=True)
    return mesh


def render_orbit_video(mesh, scalar_name, cmap, clim, output_path,
                       n_frames=360, fps=30, resolution=(1920, 1080),
                       deform_scale=0):
    """カメラ周回アニメーションを MP4 で出力する。

    Args:
        mesh: PyVista PolyData (360° フルモデル)
        scalar_name: 'smises' or 'temp'
        cmap: colormap 名
        clim: [vmin, vmax]
        output_path: 出力 MP4 パス
        n_frames: フレーム数 (360 = 1°/frame)
        fps: フレームレート
        resolution: (width, height)
        deform_scale: 変形倍率 (0 = 変形なし)
    """
    title_map = {
        'smises': 'von Mises Stress (MPa)',
        'temp': 'Temperature (°C)',
        'u_mag': 'Displacement (mm)',
    }
    display_name = title_map.get(scalar_name, scalar_name)

    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background('#1a1a2e', top='#16213e')

    render_mesh = mesh.copy()

    pl.add_mesh(
        render_mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
        scalar_bar_args={
            'title': display_name,
            'title_font_size': 18,
            'label_font_size': 14,
            'shadow': True,
            'width': 0.4,
            'height': 0.06,
            'position_x': 0.3,
            'position_y': 0.03,
            'color': 'white',
            'fmt': '%.1f',
        },
    )

    # ライティング
    pl.add_light(pv.Light(position=(20000, 15000, 20000), intensity=0.7))
    pl.add_light(pv.Light(position=(-15000, 10000, -15000), intensity=0.3))

    # タイトル
    pl.add_text(
        f'H3 Fairing — {display_name}\nLaunch Condition (360° Full Model)',
        position='upper_left', font_size=14, color='white', font='times',
    )

    # カメラ軌道パラメータ
    center = np.array([0, TOTAL_H * 0.42, 0])
    cam_dist = 22000
    elevation_base = np.radians(20)  # 仰角 20°

    print(f"  Rendering {n_frames} frames → {output_path}")
    pl.open_movie(output_path, framerate=fps, quality=8)

    for frame in range(n_frames):
        azimuth = 2 * np.pi * frame / n_frames
        # 仰角をゆっくり揺らす（20° ± 8°）
        elevation = elevation_base + np.radians(8) * np.sin(2 * np.pi * frame / n_frames)

        cam_x = cam_dist * np.cos(elevation) * np.cos(azimuth)
        cam_z = cam_dist * np.cos(elevation) * np.sin(azimuth)
        cam_y = center[1] + cam_dist * np.sin(elevation)

        pl.camera_position = [
            (cam_x, cam_y, cam_z),
            tuple(center),
            (0, 1, 0),
        ]
        pl.write_frame()

        if (frame + 1) % 60 == 0:
            print(f"    frame {frame + 1}/{n_frames}")

    pl.close()
    print(f"  Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='H3 Fairing 360° rotation video')
    parser.add_argument('--sample', type=int, default=1,
                        help='Sample number (default: 1)')
    parser.add_argument('--dataset', default='dataset_realistic_25mm_100',
                        help='Dataset directory name')
    parser.add_argument('--frames', type=int, default=360,
                        help='Number of frames (default: 360)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame rate (default: 30)')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fields', nargs='+', default=['smises', 'temp'],
                        help='Fields to render (default: smises temp)')
    args = parser.parse_args()

    sample_dir = os.path.join(
        PROJECT_ROOT, args.dataset, f'sample_{args.sample:04d}')
    out_dir = os.path.join(PROJECT_ROOT, 'figures', 'fairing_360_video')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  H3 Fairing — 360° 3D Rotation Video")
    print("=" * 60)

    # データ読み込み
    print(f"\n[1] Loading sample {args.sample:04d} ...")
    nodes, elems = load_sample(sample_dir)
    print(f"  Nodes: {len(nodes):,}, Elements: {len(elems):,}")

    # 360° フルモデル構築
    print("\n[2] Building 360° full model ...")
    coords, faces, scalars = replicate_360(nodes, elems)
    mesh = build_full_mesh(coords, faces, scalars)

    # フィールドごとに動画生成
    field_config = {
        'smises': {'cmap': 'inferno', 'clim_q': (0, 0.97)},
        'temp':   {'cmap': 'coolwarm', 'clim_q': (0.01, 0.99)},
        'u_mag':  {'cmap': 'turbo', 'clim_q': (0, 0.98)},
    }

    for field in args.fields:
        if field not in scalars:
            print(f"  [SKIP] '{field}' not found")
            continue

        print(f"\n[3] Rendering '{field}' ...")
        cfg = field_config.get(field, {'cmap': 'viridis', 'clim_q': (0, 0.98)})
        data = scalars[field]

        # パーセンタイルでカラー範囲を決定
        positive = data[data > 0.01] if field == 'smises' else data
        q_lo, q_hi = cfg['clim_q']
        vmin = np.percentile(positive, q_lo * 100)
        vmax = np.percentile(positive, q_hi * 100)
        print(f"  clim: [{vmin:.2f}, {vmax:.2f}]")

        out_path = os.path.join(
            out_dir, f'h3_fairing_360_{field}_sample{args.sample:04d}.mp4')
        render_orbit_video(
            mesh, field, cfg['cmap'], [vmin, vmax], out_path,
            n_frames=args.frames, fps=args.fps,
            resolution=(args.width, args.height),
        )

    print("\n" + "=" * 60)
    print(f"  Output directory: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
