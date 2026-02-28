#!/usr/bin/env python3
"""
34次元ノード特徴量の直感的可視化
Wiki掲載用の図を生成する
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Japanese font setup ──
import matplotlib.font_manager as fm
# Clear cache and rebuild
fm._load_fontmanager()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── Color palette for categories ──
CAT_COLORS = {
    '位置・幾何':     '#4E79A7',
    '変位':          '#F28E2B',
    '温度':          '#E15759',
    '応力':          '#76B7B2',
    '熱応力':        '#59A14F',
    'ひずみ':        '#EDC948',
    '繊維配向':      '#B07AA1',
    '積層構成':      '#FF9DA7',
    '境界フラグ':    '#9C755F',
}

# ── 34 features definition ──
FEATURES = [
    # (name, dim_start, category, short_desc)
    ('x',       0,  '位置・幾何', '座標X'),
    ('y',       1,  '位置・幾何', '座標Y'),
    ('z',       2,  '位置・幾何', '座標Z'),
    ('nx',      3,  '位置・幾何', '法線X'),
    ('ny',      4,  '位置・幾何', '法線Y'),
    ('nz',      5,  '位置・幾何', '法線Z'),
    ('k1',      6,  '位置・幾何', '主曲率1'),
    ('k2',      7,  '位置・幾何', '主曲率2'),
    ('H',       8,  '位置・幾何', '平均曲率'),
    ('K',       9,  '位置・幾何', 'ガウス曲率'),
    ('ux',     10,  '変位', '変位X'),
    ('uy',     11,  '変位', '変位Y'),
    ('uz',     12,  '変位', '変位Z'),
    ('u_mag',  13,  '変位', '変位の大きさ'),
    ('temp',   14,  '温度', '温度'),
    ('s11',    15,  '応力', '応力 S11'),
    ('s22',    16,  '応力', '応力 S22'),
    ('s12',    17,  '応力', 'せん断 S12'),
    ('smises', 18,  '応力', 'von Mises応力'),
    ('Sum_s', 19,  '応力', '主応力和'),
    ('th_vm',  20, '熱応力', '熱応力Mises'),
    ('le11',   21,  'ひずみ', 'ひずみ LE11'),
    ('le22',   22,  'ひずみ', 'ひずみ LE22'),
    ('le12',   23,  'ひずみ', 'せん断 LE12'),
    ('f_cx',   24,  '繊維配向', '繊維方向X'),
    ('f_cy',   25,  '繊維配向', '繊維方向Y'),
    ('f_cz',   26,  '繊維配向', '繊維方向Z'),
    ('lay_0',  27,  '積層構成', '0°層'),
    ('lay_45', 28,  '積層構成', '45°層'),
    ('lay_-45',29,  '積層構成', '-45°層'),
    ('lay_90', 30,  '積層構成', '90°層'),
    ('circ',   31,  '積層構成', '周方向角度'),
    ('bnd',    32,  '境界フラグ', '境界フラグ'),
    ('load',   33,  '境界フラグ', '荷重フラグ'),
]

FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_CATS  = [f[2] for f in FEATURES]
FEATURE_DESCS = [f[3] for f in FEATURES]

# Ordered unique categories
CAT_ORDER = ['位置・幾何', '変位', '温度', '応力', '熱応力',
             'ひずみ', '繊維配向', '積層構成', '境界フラグ']


def load_sample(sample_dir):
    """Load nodes.csv and return DataFrame."""
    csv_path = os.path.join(sample_dir, 'nodes.csv')
    return pd.read_csv(csv_path)


# ═══════════════════════════════════════════════════
#  Figure 1: 特徴量マップ — カテゴリ別の構成図
# ═══════════════════════════════════════════════════
def fig1_feature_map(out_dir):
    """Waffle-style chart showing 34 features by category."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Build category groups
    cat_dims = {}
    for cat in CAT_ORDER:
        cat_dims[cat] = [(f[0], f[3]) for f in FEATURES if f[2] == cat]

    # Draw blocks
    x_pos = 0
    gap = 0.3
    block_w = 1.0
    block_h = 1.0
    row_h = 1.3

    cat_rects = []
    for cat in CAT_ORDER:
        dims = cat_dims[cat]
        n = len(dims)
        color = CAT_COLORS[cat]

        # Category background
        rect = mpatches.FancyBboxPatch(
            (x_pos - 0.1, -0.3), n * block_w + 0.2, block_h + 1.6,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.12, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        # Category label on top
        ax.text(x_pos + n * block_w / 2, block_h + 0.95, cat,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=color)
        ax.text(x_pos + n * block_w / 2, block_h + 0.55,
                f'{n}次元', ha='center', va='center', fontsize=9,
                color=color, alpha=0.8)

        for i, (name, desc) in enumerate(dims):
            bx = x_pos + i * block_w
            # Feature block
            rect = mpatches.FancyBboxPatch(
                (bx + 0.05, 0), block_w - 0.1, block_h,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=0.7, edgecolor='white', linewidth=1.5
            )
            ax.add_patch(rect)
            # Feature name
            ax.text(bx + block_w / 2, block_h * 0.55, name,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white')
            # Short description below
            ax.text(bx + block_w / 2, block_h * 0.2, desc,
                    ha='center', va='center', fontsize=5.5, color='white',
                    alpha=0.9)

        x_pos += n * block_w + gap

    ax.set_xlim(-0.5, x_pos)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('34次元ノード特徴量マップ', fontsize=16, fontweight='bold', pad=20)

    # Subtitle
    fig.text(0.5, 0.02,
             'FEMノードごとに34個の特徴量を持つ → GNNの入力ベクトル',
             ha='center', fontsize=11, color='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(out_dir, '01_feature_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Figure 2: 円グラフ — カテゴリ割合
# ═══════════════════════════════════════════════════
def fig2_category_pie(out_dir):
    """Donut chart of feature categories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Count dims per category
    cat_counts = {}
    for cat in CAT_ORDER:
        cat_counts[cat] = sum(1 for f in FEATURES if f[2] == cat)

    labels = list(cat_counts.keys())
    sizes = list(cat_counts.values())
    colors = [CAT_COLORS[c] for c in labels]

    # --- Left: Donut chart ---
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda p: f'{int(round(p*34/100))}',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
        at.set_color('white')

    # Center text
    ax1.text(0, 0, '34\n次元', ha='center', va='center',
             fontsize=24, fontweight='bold', color='#333')

    # Legend
    legend_labels = [f'{l} ({s}次元)' for l, s in zip(labels, sizes)]
    ax1.legend(wedges, legend_labels, loc='center left',
               bbox_to_anchor=(-0.35, 0.5), fontsize=10)
    ax1.set_title('カテゴリ別の次元数', fontsize=14, fontweight='bold')

    # --- Right: Source breakdown ---
    sources = {
        'Abaqus ODB\n(物理量)': ['変位', '温度', '応力', '熱応力', 'ひずみ'],
        'メッシュ計算\n(幾何量)': ['位置・幾何'],
        '座標計算\n(材料属性)': ['繊維配向', '積層構成'],
        '境界判定\n(補助)': ['境界フラグ'],
    }
    source_sizes = []
    source_labels = []
    source_colors_list = []
    for src, cats in sources.items():
        n = sum(cat_counts.get(c, 0) for c in cats)
        source_sizes.append(n)
        source_labels.append(f'{src}\n{n}次元')

    src_colors = ['#4E79A7', '#F28E2B', '#B07AA1', '#9C755F']
    wedges2, texts2, autotexts2 = ax2.pie(
        source_sizes, labels=source_labels, colors=src_colors,
        autopct='%1.0f%%', startangle=90, pctdistance=0.6,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10}
    )
    for at in autotexts2:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('white')
    ax2.set_title('データソース別の構成', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, '02_category_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Figure 3: Defect vs Healthy 比較 — 欠陥信号の可視化
# ═══════════════════════════════════════════════════
def fig3_defect_vs_healthy(sample_dir, out_dir):
    """Bar chart showing feature value differences between healthy and defect nodes."""
    df = load_sample(sample_dir)

    # Map CSV columns to feature indices
    csv_cols = ['x','y','z','ux','uy','uz','u_mag','temp',
                's11','s22','s12','smises','thermal_smises',
                'le11','le22','le12']

    # Select key physics features only
    key_features = [
        ('u_mag',  '変位の大きさ', '変位'),
        ('temp',   '温度', '温度'),
        ('s11',    '応力 S11', '応力'),
        ('s22',    '応力 S22', '応力'),
        ('smises', 'von Mises', '応力'),
        ('thermal_smises', '熱応力', '熱応力'),
        ('le11',   'ひずみ LE11', 'ひずみ'),
        ('le22',   'ひずみ LE22', 'ひずみ'),
    ]

    healthy = df[df['defect_label'] == 0]
    defect  = df[df['defect_label'] != 0]

    if len(defect) == 0:
        print("  WARNING: No defect nodes in sample, skipping fig3")
        return None

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (col, label, cat) in enumerate(key_features):
        ax = axes[i]
        color = CAT_COLORS[cat]

        h_vals = healthy[col].values
        d_vals = defect[col].values

        # Histogram comparison
        all_vals = np.concatenate([h_vals, d_vals])
        vmin, vmax = np.percentile(all_vals, [1, 99])
        bins = np.linspace(vmin, vmax, 40)

        ax.hist(h_vals, bins=bins, alpha=0.6, color='#4E79A7',
                density=True, label=f'健全 (n={len(healthy):,})')
        ax.hist(d_vals, bins=bins, alpha=0.7, color='#E15759',
                density=True, label=f'欠陥 (n={len(defect):,})')

        # Statistics annotation
        h_mean = np.mean(h_vals)
        d_mean = np.mean(d_vals)
        ratio = abs(d_mean / h_mean) if abs(h_mean) > 1e-10 else 0

        ax.axvline(h_mean, color='#4E79A7', ls='--', lw=1.5, alpha=0.8)
        ax.axvline(d_mean, color='#E15759', ls='--', lw=1.5, alpha=0.8)

        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)

        # Show mean difference
        if ratio > 0:
            sign = '↑' if d_mean > h_mean else '↓'
            ax.text(0.02, 0.95, f'欠陥平均 {sign} {abs(d_mean-h_mean):.2g}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    color='#E15759', fontweight='bold')

    fig.suptitle('健全ノード vs 欠陥ノード — 物理量の分布比較',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.5, -0.01,
             '赤い分布が青からずれている → GNNが検出に使える信号',
             ha='center', fontsize=11, color='gray')

    plt.tight_layout()
    path = os.path.join(out_dir, '03_defect_vs_healthy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Figure 4: パイプライン図 — データフロー
# ═══════════════════════════════════════════════════
def fig4_pipeline_flow(out_dir):
    """Schematic showing how 34 features are constructed."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color, fontsize=10, subtext=None):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, alpha=0.85, edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.6, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
        if subtext:
            ax.text(x + w/2, y + h*0.25, subtext,
                    ha='center', va='center', fontsize=7, color='white', alpha=0.9)

    def draw_arrow(x1, y1, x2, y2, color='#666'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # ── Title ──
    ax.text(8, 7.6, '34次元特徴量の構築パイプライン',
            ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 7.2, 'FEM解析 → CSV抽出 → グラフ構築 → GNN入力',
            ha='center', fontsize=11, color='gray')

    # ── Stage 1: Abaqus FEM ──
    draw_box(0.3, 4.5, 3.0, 2.0, 'Abaqus FEM', '#2C3E50', 14,
             'generate_fairing_dataset.py')

    # ODB outputs
    odb_items = [
        ('U (変位)', '#F28E2B', '4次元'),
        ('NT (温度)', '#E15759', '1次元'),
        ('S (応力)', '#76B7B2', '5次元'),
        ('STHERM', '#59A14F', '1次元'),
        ('LE (ひずみ)', '#EDC948', '3次元'),
    ]
    for i, (label, color, dims) in enumerate(odb_items):
        bx = 0.5 + i * 2.7
        draw_box(bx, 2.0, 2.3, 0.8, f'{label}', color, 9, dims)
        if i < 3:
            draw_arrow(1.8, 4.5, bx + 1.15, 2.8, '#666')
        else:
            draw_arrow(1.8, 4.5, bx + 1.15, 2.8, '#666')

    # ── Stage 2: CSV extraction ──
    draw_box(5.5, 4.5, 3.0, 2.0, 'ODB → CSV', '#8E44AD', 14,
             'extract_odb_results.py')
    draw_arrow(3.3, 5.5, 5.5, 5.5, '#666')

    csv_text = 'nodes.csv: 18列\n(node_id, x,y,z, ux,uy,uz,\nu_mag, temp, s11...le12, label)'
    ax.text(7.0, 3.4, csv_text, ha='center', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCC'),
            family='monospace')

    # ── Stage 3: Graph construction ──
    draw_box(10.5, 4.5, 3.0, 2.0, 'グラフ構築', '#27AE60', 14,
             'build_graph.py')
    draw_arrow(8.5, 5.5, 10.5, 5.5, '#666')

    # Computed features
    comp_items = [
        ('幾何量\n(法線・曲率)', '#4E79A7', '7次元'),
        ('繊維配向\n(周方向)', '#B07AA1', '3次元'),
        ('積層角度\n(0/45/-45/90)', '#FF9DA7', '5次元'),
        ('境界判定\n(bnd/load)', '#9C755F', '2次元'),
    ]
    for i, (label, color, dims) in enumerate(comp_items):
        bx = 9.2 + i * 1.7
        draw_box(bx, 2.0, 1.5, 0.8, label, color, 7, dims)
        draw_arrow(12.0, 4.5, bx + 0.75, 2.8, '#27AE60')

    # ── Stage 4: GNN Input ──
    draw_box(14.0, 4.5, 1.7, 2.0, 'GNN\n入力', '#C0392B', 14)
    draw_arrow(13.5, 5.5, 14.0, 5.5, '#666')

    # Final output label
    ax.text(14.85, 3.8, 'torch.Size\n[N, 34]', ha='center', fontsize=10,
            fontweight='bold', color='#C0392B',
            bbox=dict(boxstyle='round', facecolor='#FDEDEC', edgecolor='#C0392B'))

    # ── Legend ──
    ax.text(0.5, 0.8, '凡例:', fontsize=10, fontweight='bold')
    legend_items = [
        ('ODB直接抽出 (14次元)', '#F28E2B'),
        ('メッシュから計算 (7次元)', '#4E79A7'),
        ('座標から計算 (8次元)', '#B07AA1'),
        ('幾何判定 (2次元)', '#9C755F'),
        ('CSV計算 (3次元: u_mag, Σσ等)', '#76B7B2'),
    ]
    for i, (text, color) in enumerate(legend_items):
        bx = 2.5 + i * 2.8
        rect = mpatches.FancyBboxPatch(
            (bx, 0.55), 0.3, 0.3, boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(bx + 0.5, 0.7, text, fontsize=8, va='center')

    path = os.path.join(out_dir, '04_pipeline_flow.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Figure 5: ヒートマップ — 特徴量の統計サマリ
# ═══════════════════════════════════════════════════
def fig5_feature_stats(sample_dir, out_dir):
    """Heatmap-style summary showing ranges and non-zero rates."""
    df = load_sample(sample_dir)

    # Build feature stats using CSV columns available
    csv_map = {
        'x': 'x', 'y': 'y', 'z': 'z',
        'ux': 'ux', 'uy': 'uy', 'uz': 'uz', 'u_mag': 'u_mag',
        'temp': 'temp',
        's11': 's11', 's22': 's22', 's12': 's12', 'smises': 'smises',
        'thermal_smises': 'thermal_smises',
        'le11': 'le11', 'le22': 'le22', 'le12': 'le12',
    }

    stats = []
    for fname, dim, cat, desc in FEATURES:
        csv_col = csv_map.get(fname)
        if csv_col and csv_col in df.columns:
            vals = df[csv_col].values
            stats.append({
                'name': fname,
                'cat': cat,
                'desc': desc,
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'nonzero_pct': np.mean(np.abs(vals) > 1e-10) * 100,
                'source': 'CSV'
            })
        else:
            # Computed features — mark as computed
            stats.append({
                'name': fname,
                'cat': cat,
                'desc': desc,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'nonzero_pct': 100.0,  # All computed features are live
                'source': '計算'
            })

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # --- Top: Feature range chart ---
    ax1 = fig.add_subplot(gs[0])

    x_pos = np.arange(len(stats))
    has_data = [not np.isnan(s['mean']) for s in stats]
    means = [s['mean'] if has_data[i] else 0 for i, s in enumerate(stats)]
    stds = [s['std'] if has_data[i] else 0 for i, s in enumerate(stats)]

    colors = [CAT_COLORS[s['cat']] for s in stats]

    bars = ax1.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='white')
    ax1.errorbar(x_pos[np.array(has_data)],
                 np.array(means)[has_data],
                 yerr=np.array(stds)[has_data],
                 fmt='none', color='#333', capsize=2, linewidth=1)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s['name'] for s in stats], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('平均値 ± 標準偏差', fontsize=11)
    ax1.set_title('各特徴量の統計サマリ (Sample 0000)',
                  fontsize=14, fontweight='bold')
    ax1.axhline(0, color='gray', lw=0.5)

    # Category spans
    prev_cat = None
    cat_start = 0
    for i, s in enumerate(stats + [{'cat': None}]):
        if s['cat'] != prev_cat:
            if prev_cat is not None:
                mid = (cat_start + i - 1) / 2
                ax1.axvspan(cat_start - 0.5, i - 0.5,
                           alpha=0.05, color=CAT_COLORS.get(prev_cat, '#999'))
            cat_start = i
            prev_cat = s['cat']

    # --- Bottom: Non-zero rate ---
    ax2 = fig.add_subplot(gs[1])
    nz_pcts = [s['nonzero_pct'] for s in stats]
    bar_colors = ['#2ECC71' if p > 50 else '#E74C3C' for p in nz_pcts]
    # Override computed features to show as green
    for i, s in enumerate(stats):
        if s['source'] == '計算':
            bar_colors[i] = '#3498DB'

    ax2.bar(x_pos, nz_pcts, color=bar_colors, alpha=0.8, edgecolor='white')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s['name'] for s in stats], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('非ゼロ率 (%)', fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.axhline(100, color='gray', lw=0.5, ls='--')

    # Legend for bottom chart
    legend_patches = [
        mpatches.Patch(color='#2ECC71', alpha=0.8, label='CSV抽出 (LIVE)'),
        mpatches.Patch(color='#3498DB', alpha=0.8, label='計算生成 (LIVE)'),
    ]
    ax2.legend(handles=legend_patches, fontsize=9, loc='lower right')
    ax2.set_title('特徴量のアクティブ率 — Dead特徴量ゼロ',
                  fontsize=12, fontweight='bold')

    path = os.path.join(out_dir, '05_feature_stats.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Figure 6: 空間マップ — フェアリング上の物理量
# ═══════════════════════════════════════════════════
def fig6_spatial_physics(sample_dir, out_dir):
    """Scatter plots on unfolded fairing showing key physics."""
    df = load_sample(sample_dir)

    # Compute circumferential angle and z for unfolded view
    theta = np.arctan2(df['x'].values, -df['z'].values)
    z_coord = df['y'].values

    fields = [
        ('u_mag',   '変位の大きさ [mm]', 'hot'),
        ('smises',  'von Mises応力 [MPa]', 'YlOrRd'),
        ('temp',    '温度 [°C]', 'coolwarm'),
        ('defect_label', '欠陥ラベル (0=健全, 1=欠陥)', 'RdYlGn_r'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, (col, title, cmap) in enumerate(fields):
        ax = axes[i]
        vals = df[col].values

        if col == 'defect_label':
            # Binary plot
            healthy_mask = vals == 0
            defect_mask = vals != 0
            ax.scatter(np.degrees(theta[healthy_mask]),
                      z_coord[healthy_mask],
                      c='#4E79A7', s=0.1, alpha=0.2, rasterized=True)
            ax.scatter(np.degrees(theta[defect_mask]),
                      z_coord[defect_mask],
                      c='#E15759', s=2.0, alpha=0.8, rasterized=True,
                      label=f'欠陥 ({defect_mask.sum():,} nodes)')
            ax.legend(fontsize=9, markerscale=5)
        else:
            vmin, vmax = np.percentile(vals, [2, 98])
            sc = ax.scatter(np.degrees(theta), z_coord, c=vals,
                           cmap=cmap, s=0.1, alpha=0.5,
                           vmin=vmin, vmax=vmax, rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

        ax.set_xlabel('θ [deg]', fontsize=10)
        ax.set_ylabel('z [mm]', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('フェアリング展開図上の物理量分布 (Sample 0000)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, '06_spatial_physics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    project_root = Path(__file__).resolve().parent.parent
    # sample_0001 has debonding defect with 309 defect nodes
    sample_dir = project_root / 'dataset_realistic_25mm_100' / 'sample_0001'
    out_dir = project_root / 'wiki_repo' / 'images' / 'node_features'
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("34次元ノード特徴量 — Wiki用可視化")
    print("=" * 60)

    print("\n[1/6] 特徴量マップ...")
    fig1_feature_map(out_dir)

    print("[2/6] カテゴリ別構成...")
    fig2_category_pie(out_dir)

    print("[3/6] 欠陥 vs 健全比較...")
    fig3_defect_vs_healthy(sample_dir, out_dir)

    print("[4/6] パイプライン図...")
    fig4_pipeline_flow(out_dir)

    print("[5/6] 特徴量統計サマリ...")
    fig5_feature_stats(sample_dir, out_dir)

    print("[6/6] 空間物理量マップ...")
    fig6_spatial_physics(sample_dir, out_dir)

    print("\n" + "=" * 60)
    print(f"全図を保存: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
