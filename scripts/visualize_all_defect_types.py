#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全欠陥タイプの概念可視化 — All Defect Types Visualization

7種類の欠陥（debonding, fod, impact, delamination, inner_debond,
thermal_progression, acoustic_fatigue）+ クラック・コア損傷の概念図を生成。
静的 FEM の応力・変位パターンを模擬したシグネチャで表現。

出力:
  - defect_*_comparison.png: 2D contour (Healthy/Defective/Residual)
  - defect_*_3d.png: 3D 円筒フェアリング上の欠陥シグネチャ（画像風）
  - defect_*_animation.gif: 欠陥シグネチャの時間変化（概念アニメーション）

Usage:
  python scripts/visualize_all_defect_types.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "wiki_repo/images/defects"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Geometry
R = 2600.0
H_CYL = 5000.0
H_NOSE = 5400.0
H_TOTAL = H_CYL + H_NOSE
theta = np.linspace(0, np.pi/3, 120)
z_cyl = np.linspace(0, H_CYL, 120)
z_nose = np.linspace(H_CYL, H_TOTAL, 80)
T_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
T_nose, Z_nose = np.meshgrid(theta, z_nose)
rho = (R**2 + H_NOSE**2) / (2*R)
xc = R - rho
R_nose = np.sqrt(np.maximum(rho**2 - (Z_nose - H_CYL)**2, 0)) + xc
T = np.vstack((T_cyl, T_nose))
Z = np.vstack((Z_cyl, Z_nose))
Arc = T * R


def add_debonding(field, t_c, z_c, radius=200, amp=0.8):
    """外スキン-コア剥離: 滑らかな膨らみ (荷重伝達喪失)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    return field + amp * np.exp(-dist_sq / (2 * radius**2))


def add_fod(field, t_c, z_c, radius=80, amp=25):
    """FOD: 局所的な応力集中 (硬質異物)"""
    dist = np.sqrt((R * (T - t_c))**2 + (Z - z_c)**2)
    return field + amp * np.exp(-dist / (0.25 * radius))


def add_impact(field, t_c, z_c, radius=120, amp=-0.6):
    """衝撃損傷: 中央凹み + 周囲盛り上がり (BVID)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    sigma = radius / 2.0
    norm_sq = dist_sq / (sigma**2)
    return field + amp * (1 - norm_sq) * np.exp(-norm_sq / 2)


def add_delamination(field, t_c, z_c, radius=150, amp=0.4):
    """デラミネーション: 積層内剥離、せん断剛性低下 (やや広域・弱い)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    return field + amp * np.exp(-dist_sq / (3 * radius**2))


def add_crack(field, t_c, z_c, angle=0.3, length=250, width=60, amp=8):
    """クラック: 線状の応力集中 (マトリクス割れ)"""
    dx = R * (T - t_c) * np.cos(angle) + (Z - z_c) * np.sin(angle)
    dy = -R * (T - t_c) * np.sin(angle) + (Z - z_c) * np.cos(angle)
    along = np.abs(dx)
    across = np.abs(dy)
    mask = (along < length) & (across < width)
    pert = amp * np.exp(-across**2 / (width * 0.5)**2) * np.maximum(0, 1 - along / length) * mask
    return field + pert


def add_inner_debond(field, t_c, z_c, radius=180, amp=0.35):
    """内スキン-コア剥離: 外表面への影響は弱い (内界面)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    return field + amp * np.exp(-dist_sq / (2.5 * radius**2))


def add_thermal_progression(field, t_c, z_c, radius=250, amp=0.3):
    """熱応力進展: 広域・低振幅 (CTE 不整合)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    return field + amp * np.exp(-dist_sq / (4 * radius**2))


def add_acoustic_fatigue(field, t_c, z_c, radius=120, amp=0.25):
    """音響疲労: 局所・低振幅 (界面疲労)"""
    dist_sq = (R * (T - t_c))**2 + (Z - z_c)**2
    return field + amp * np.exp(-dist_sq / (2 * radius**2))


def add_core_damage(field, t_c, z_c, radius=100, amp=-0.4):
    """コア損傷: コア圧潰による凹み (impact に類似)"""
    return add_impact(field, t_c, z_c, radius, amp)


# Base fields
def make_base_disp():
    return 3.0 * np.sin(np.pi * Z / H_TOTAL * 0.8) * (1 + 0.2*np.cos(3*T)) + \
           1.5 * (Z / H_TOTAL)**2 * np.cos(T)

def make_base_stress():
    return 75.0 * ((H_TOTAL - Z) / H_TOTAL)**1.5 + 5.0 * np.cos(3 * T) * (Z / H_TOTAL)


DEFECT_CONFIGS = [
    ('debonding', 'Debonding (outer skin-core)', add_debonding, (np.pi/6, 2500), 'displacement'),
    ('fod', 'FOD (foreign object)', add_fod, (np.pi/4, 3500), 'stress'),
    ('impact', 'Impact (BVID)', add_impact, (np.pi/12, 6000), 'displacement'),
    ('delamination', 'Delamination (inter-ply)', add_delamination, (np.pi/5, 2000), 'displacement'),
    ('crack', 'Crack (matrix)', add_crack, (np.pi/8, 4000), 'stress'),
    ('inner_debond', 'Inner Debond', add_inner_debond, (np.pi/7, 3000), 'displacement'),
    ('thermal', 'Thermal Progression', add_thermal_progression, (np.pi/6, 1500), 'displacement'),
    ('acoustic', 'Acoustic Fatigue', add_acoustic_fatigue, (np.pi/9, 4500), 'displacement'),
    ('core_damage', 'Core Damage', add_core_damage, (np.pi/10, 5000), 'displacement'),
]


def plot_single_defect(defect_id, name, add_fn, pos, field_type):
    """単一欠陥の 3 パネル (Healthy / Defective / Residual)"""
    t_c, z_c = pos
    if field_type == 'displacement':
        base = make_base_disp()
    else:
        base = make_base_stress()
    defective = add_fn(base.copy(), t_c, z_c)
    residual = defective - base

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = 'viridis' if field_type == 'displacement' else 'hot'
    for ax, data, title in [(axes[0], base, 'Healthy'), (axes[1], defective, 'Defective'), (axes[2], residual, 'Residual')]:
        if 'Residual' in title:
            lim = np.max(np.abs(residual)) or 1
            im = ax.contourf(Arc, Z, data, 80, cmap='RdBu_r', vmin=-lim, vmax=lim)
        else:
            im = ax.contourf(Arc, Z, data, 80, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Arc [mm]')
        ax.set_ylabel('Z [mm]')
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(name.replace('\n', ' '), fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'defect_{defect_id}_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', path)


def plot_defect_3d(defect_id, name, add_fn, pos, field_type):
    """3D 円筒フェアリング上に欠陥シグネチャをマッピング（画像風）"""
    t_c, z_c = pos
    if field_type == 'displacement':
        base = make_base_disp()
    else:
        base = make_base_stress()
    defective = add_fn(base.copy(), t_c, z_c)
    residual = defective - base

    # 円筒座標 (theta, z) -> (x, y, z) for 30° sector
    n_theta, n_z = 80, 100
    theta_1d = np.linspace(0, np.pi/3, n_theta)
    z_1d = np.linspace(0, H_CYL, n_z)
    Theta, Z_grid = np.meshgrid(theta_1d, z_1d)
    X = (R + CORE_T/2) * np.cos(Theta)
    Y = Z_grid
    Z_3d = (R + CORE_T/2) * np.sin(Theta)

    # グリッド上の residual を補間
    from scipy.interpolate import griddata
    pts = np.column_stack([T.ravel(), Z.ravel()])
    vals = residual.ravel()
    res_interp = griddata(pts, vals, (Theta, Z_grid), method='cubic', fill_value=0)

    # 正規化してカラーマップ
    vmax = np.max(np.abs(res_interp)) or 1
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    colors = cm.RdBu_r(norm(res_interp))
    colors = np.clip(colors, 0, 1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_3d, facecolors=colors, rstride=1, cstride=1,
                    shade=False, antialiased=True, alpha=0.95)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y (axial) [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('%s — Defect Signature on Fairing' % name, fontsize=13, fontweight='bold')
    ax.view_init(elev=18, azim=-55)
    m = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
    m.set_array(res_interp)
    fig.colorbar(m, ax=ax, shrink=0.6, label='Residual')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'defect_{defect_id}_3d.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved:', path)


# Cylinder radius for 3D (R + half core)
CORE_T = 38.0


def plot_overview_grid():
    """全欠陥の Residual を 3x3 グリッドで一覧"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (defect_id, name, add_fn, pos, field_type) in enumerate(DEFECT_CONFIGS):
        t_c, z_c = pos
        base = make_base_disp() if field_type == 'displacement' else make_base_stress()
        defective = add_fn(base.copy(), t_c, z_c)
        residual = defective - base

        ax = axes[idx]
        lim = np.max(np.abs(residual)) or 1
        im = ax.contourf(Arc, Z, residual, 60, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Arc [mm]')
        ax.set_ylabel('Z [mm]')
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.suptitle('Defect Signatures (Residual: Defective - Healthy)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'defect_types_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', path)


def plot_defect_animation(defect_id, name, add_fn, pos, field_type, n_frames=24, fps=12):
    """欠陥シグネチャの時間変化を GIF で出力（概念アニメーション）

    Residual（Defective - Healthy）を表示。欠陥がはっきり見えるようにする。
    """
    t_c, z_c = pos
    if field_type == 'displacement':
        base = make_base_disp()
    else:
        base = make_base_stress()
    defective = add_fn(base.copy(), t_c, z_c)
    residual = defective - base

    vmax = np.max(np.abs(residual)) or 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    def animate(i):
        ax.clear()
        t_frac = (i + 1) / n_frames
        # Residual を徐々に表示（0→1 でフェードイン）
        frame_data = residual * t_frac
        im = ax.contourf(Arc, Z, frame_data, 60, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlim(Arc.min(), Arc.max())
        ax.set_ylim(Z.min(), Z.max())
        ax.set_xlabel('Arc [mm]')
        ax.set_ylabel('Z [mm]')
        ax.set_title('%s — Residual (t = %.0f%%)' % (name, t_frac * 100), fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Residual')
        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False)
    path = os.path.join(OUTPUT_DIR, f'defect_{defect_id}_animation.gif')
    anim.save(path, writer='pillow', fps=fps, dpi=80)
    plt.close()
    print('Saved:', path)


def plot_overview_3d():
    """全欠陥の 3D ビューを 3x3 グリッドで一覧"""
    from scipy.interpolate import griddata
    fig = plt.figure(figsize=(18, 16))
    n_theta, n_z = 40, 50
    theta_1d = np.linspace(0, np.pi/3, n_theta)
    z_1d = np.linspace(0, H_CYL, n_z)
    Theta, Z_grid = np.meshgrid(theta_1d, z_1d)
    X = (R + CORE_T/2) * np.cos(Theta)
    Y = Z_grid
    Z_3d = (R + CORE_T/2) * np.sin(Theta)
    pts = np.column_stack([T.ravel(), Z.ravel()])

    for idx, (defect_id, name, add_fn, pos, field_type) in enumerate(DEFECT_CONFIGS):
        t_c, z_c = pos
        base = make_base_disp() if field_type == 'displacement' else make_base_stress()
        defective = add_fn(base.copy(), t_c, z_c)
        residual = defective - base
        vals = residual.ravel()
        res_interp = griddata(pts, vals, (Theta, Z_grid), method='cubic', fill_value=0)
        vmax = np.max(np.abs(res_interp)) or 1
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        colors = cm.RdBu_r(norm(res_interp))

        ax = fig.add_subplot(3, 3, idx + 1, projection='3d')
        ax.plot_surface(X, Y, Z_3d, facecolors=colors, rstride=1, cstride=1,
                        shade=False, alpha=0.95)
        ax.set_title(name, fontsize=9)
        ax.view_init(elev=20, azim=-50)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.suptitle('Defect Signatures on Fairing (3D)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'defect_types_overview_3d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', path)


def main():
    print('Generating defect type visualizations...')
    for cfg in DEFECT_CONFIGS:
        plot_single_defect(*cfg)
        plot_defect_3d(*cfg)
        plot_defect_animation(*cfg)
    plot_overview_grid()
    plot_overview_3d()
    print('Done.')


if __name__ == '__main__':
    main()
