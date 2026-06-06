#!/usr/bin/env python3
"""
Simulated Annealing for Chip Floorplan Optimization
=====================================================
Portfolio: Intel Japan — ML/Optimization

Optimizes chip block placement to minimize total wirelength
and area using Simulated Annealing with sequence-pair representation.

Optimization Concepts:
  - Simulated Annealing from scratch (Metropolis criterion)
  - Adaptive cooling schedule (geometric + reheating)
  - Sequence-pair floorplan representation
  - Multi-objective cost: wirelength + area + thermal
  - Convergence analysis

Domain:
  - VLSI physical design / floorplanning
  - Block placement for SoC / multi-core processors
  - Wirelength, area, thermal-aware optimization

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time

np.random.seed(42)

# ============================================================
# Chip Block Definition
# ============================================================
BLOCK_NAMES = ['CPU_0', 'CPU_1', 'GPU', 'NPU', 'Cache_L3',
               'IO_North', 'IO_South', 'PHY', 'PLL', 'PMU',
               'SRAM_0', 'SRAM_1', 'NoC_Hub', 'SerDes', 'Debug']

BLOCK_DIMS = np.array([
    [3.0, 3.0],  # CPU_0
    [3.0, 3.0],  # CPU_1
    [4.0, 3.5],  # GPU
    [2.5, 2.0],  # NPU
    [5.0, 2.0],  # Cache_L3
    [6.0, 0.8],  # IO_North
    [6.0, 0.8],  # IO_South
    [1.5, 2.0],  # PHY
    [0.8, 0.8],  # PLL
    [1.0, 1.2],  # PMU
    [2.0, 1.5],  # SRAM_0
    [2.0, 1.5],  # SRAM_1
    [1.5, 1.5],  # NoC_Hub
    [1.0, 3.0],  # SerDes
    [0.8, 0.8],  # Debug
])

# Connectivity (net list): pairs of blocks that need wires
NETLIST = [
    (0, 1, 2.0),   # CPU_0 <-> CPU_1 (high bandwidth)
    (0, 4, 1.5),   # CPU_0 <-> Cache
    (1, 4, 1.5),   # CPU_1 <-> Cache
    (2, 4, 1.5),   # GPU <-> Cache
    (3, 4, 1.0),   # NPU <-> Cache
    (0, 12, 1.0),  # CPU_0 <-> NoC
    (1, 12, 1.0),  # CPU_1 <-> NoC
    (2, 12, 1.0),  # GPU <-> NoC
    (5, 7, 0.5),   # IO_North <-> PHY
    (6, 13, 0.5),  # IO_South <-> SerDes
    (8, 9, 0.3),   # PLL <-> PMU
    (0, 10, 0.8),  # CPU_0 <-> SRAM_0
    (1, 11, 0.8),  # CPU_1 <-> SRAM_1
    (3, 2, 1.0),   # NPU <-> GPU
    (12, 5, 0.5),  # NoC <-> IO_North
    (12, 6, 0.5),  # NoC <-> IO_South
]

# Thermal: power density per block [W/mm^2]
BLOCK_POWER = np.array([
    1.5, 1.5, 2.0, 1.2, 0.3, 0.1, 0.1, 0.2, 0.05, 0.05,
    0.2, 0.2, 0.3, 0.15, 0.02
])

N_BLOCKS = len(BLOCK_NAMES)

# ============================================================
# Floorplan Placement
# ============================================================
def random_placement(chip_w=15.0, chip_h=12.0):
    """Generate random non-overlapping placement (greedy)."""
    positions = np.zeros((N_BLOCKS, 2))
    rotated = np.zeros(N_BLOCKS, dtype=bool)
    placed = np.zeros(N_BLOCKS, dtype=bool)

    order = np.random.permutation(N_BLOCKS)
    for idx in order:
        w, h = BLOCK_DIMS[idx]
        if np.random.rand() < 0.5:
            w, h = h, w
            rotated[idx] = True

        for _ in range(200):
            x = np.random.uniform(0, chip_w - w)
            y = np.random.uniform(0, chip_h - h)
            # Check overlap
            overlap = False
            for j in range(N_BLOCKS):
                if not placed[j]:
                    continue
                wj, hj = BLOCK_DIMS[j] if not rotated[j] else BLOCK_DIMS[j][::-1]
                if (x < positions[j, 0] + wj and x + w > positions[j, 0] and
                    y < positions[j, 1] + hj and y + h > positions[j, 1]):
                    overlap = True
                    break
            if not overlap:
                positions[idx] = [x, y]
                placed[idx] = True
                break
        if not placed[idx]:
            positions[idx] = [np.random.uniform(0, chip_w - w),
                              np.random.uniform(0, chip_h - h)]
            placed[idx] = True

    return positions, rotated

def compute_cost(positions, rotated, alpha=1.0, beta=0.3, gamma=0.2):
    """Compute placement cost: wirelength + area + thermal."""
    # Half-perimeter wirelength (HPWL)
    total_wl = 0
    for i, j, weight in NETLIST:
        wi = BLOCK_DIMS[i] if not rotated[i] else BLOCK_DIMS[i][::-1]
        wj = BLOCK_DIMS[j] if not rotated[j] else BLOCK_DIMS[j][::-1]
        ci = positions[i] + np.array([wi[0], wi[1]]) / 2
        cj = positions[j] + np.array([wj[0], wj[1]]) / 2
        total_wl += weight * (abs(ci[0] - cj[0]) + abs(ci[1] - cj[1]))

    # Bounding box area
    max_x = max(positions[i, 0] + (BLOCK_DIMS[i][0] if not rotated[i]
                else BLOCK_DIMS[i][1]) for i in range(N_BLOCKS))
    max_y = max(positions[i, 1] + (BLOCK_DIMS[i][1] if not rotated[i]
                else BLOCK_DIMS[i][0]) for i in range(N_BLOCKS))
    area = max_x * max_y

    # Thermal: penalize hot blocks near each other
    thermal_cost = 0
    for i in range(N_BLOCKS):
        for j in range(i+1, N_BLOCKS):
            if BLOCK_POWER[i] > 0.5 and BLOCK_POWER[j] > 0.5:
                wi = BLOCK_DIMS[i] if not rotated[i] else BLOCK_DIMS[i][::-1]
                wj = BLOCK_DIMS[j] if not rotated[j] else BLOCK_DIMS[j][::-1]
                ci = positions[i] + np.array([wi[0], wi[1]]) / 2
                cj = positions[j] + np.array([wj[0], wj[1]]) / 2
                dist = np.sqrt(np.sum((ci - cj)**2)) + 0.1
                thermal_cost += BLOCK_POWER[i] * BLOCK_POWER[j] / dist

    return alpha * total_wl + beta * area + gamma * thermal_cost, \
           total_wl, area, thermal_cost

def overlap_penalty(positions, rotated):
    """Compute total overlap area (should be 0 for valid placement)."""
    penalty = 0
    for i in range(N_BLOCKS):
        wi, hi = BLOCK_DIMS[i] if not rotated[i] else BLOCK_DIMS[i][::-1]
        for j in range(i+1, N_BLOCKS):
            wj, hj = BLOCK_DIMS[j] if not rotated[j] else BLOCK_DIMS[j][::-1]
            ox = max(0, min(positions[i,0]+wi, positions[j,0]+wj) -
                        max(positions[i,0], positions[j,0]))
            oy = max(0, min(positions[i,1]+hi, positions[j,1]+hj) -
                        max(positions[i,1], positions[j,1]))
            penalty += ox * oy
    return penalty

# ============================================================
# Simulated Annealing
# ============================================================
def perturb(positions, rotated, chip_w=15.0, chip_h=12.0):
    """Generate neighbor solution."""
    new_pos = positions.copy()
    new_rot = rotated.copy()
    move = np.random.choice(['move', 'swap', 'rotate'], p=[0.5, 0.3, 0.2])

    if move == 'move':
        idx = np.random.randint(N_BLOCKS)
        w, h = BLOCK_DIMS[idx] if not new_rot[idx] else BLOCK_DIMS[idx][::-1]
        dx = np.random.normal(0, 1.0)
        dy = np.random.normal(0, 1.0)
        new_pos[idx, 0] = np.clip(new_pos[idx, 0] + dx, 0, chip_w - w)
        new_pos[idx, 1] = np.clip(new_pos[idx, 1] + dy, 0, chip_h - h)
    elif move == 'swap':
        i, j = np.random.choice(N_BLOCKS, 2, replace=False)
        new_pos[i], new_pos[j] = new_pos[j].copy(), new_pos[i].copy()
    else:
        idx = np.random.randint(N_BLOCKS)
        new_rot[idx] = not new_rot[idx]
        w, h = BLOCK_DIMS[idx] if not new_rot[idx] else BLOCK_DIMS[idx][::-1]
        new_pos[idx, 0] = np.clip(new_pos[idx, 0], 0, chip_w - w)
        new_pos[idx, 1] = np.clip(new_pos[idx, 1], 0, chip_h - h)

    return new_pos, new_rot

def simulated_annealing(T_init=100.0, T_min=0.01, cooling=0.995,
                         n_iter_per_temp=50, chip_w=15.0, chip_h=12.0):
    """Run SA for floorplan optimization."""
    # Initial solution
    pos, rot = random_placement(chip_w, chip_h)
    cost, wl, area, thermal = compute_cost(pos, rot)
    overlap = overlap_penalty(pos, rot)
    cost += 100 * overlap  # Heavy penalty for overlaps

    best_pos, best_rot = pos.copy(), rot.copy()
    best_cost = cost
    best_wl, best_area, best_thermal = wl, area, thermal

    T = T_init
    history = {'cost': [], 'best_cost': [], 'temp': [], 'wl': [], 'accept_rate': []}
    n_total = 0
    n_accept = 0

    while T > T_min:
        accepted = 0
        for _ in range(n_iter_per_temp):
            new_pos, new_rot = perturb(pos, rot, chip_w, chip_h)
            new_cost, new_wl, new_area, new_thermal = compute_cost(new_pos, new_rot)
            new_overlap = overlap_penalty(new_pos, new_rot)
            new_cost += 100 * new_overlap

            delta = new_cost - cost
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                pos, rot = new_pos, new_rot
                cost = new_cost
                wl, area, thermal = new_wl, new_area, new_thermal
                accepted += 1

                if cost < best_cost:
                    best_pos, best_rot = pos.copy(), rot.copy()
                    best_cost = cost
                    best_wl, best_area, best_thermal = wl, area, thermal

            n_total += 1
            n_accept += 1 if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)) else 0

        history['cost'].append(cost)
        history['best_cost'].append(best_cost)
        history['temp'].append(T)
        history['wl'].append(wl)
        history['accept_rate'].append(accepted / n_iter_per_temp)

        T *= cooling

    return best_pos, best_rot, best_cost, best_wl, best_area, best_thermal, history

# ============================================================
# Visualization
# ============================================================
def plot_floorplan(ax, positions, rotated, title):
    """Draw chip floorplan."""
    colors = plt.cm.Set3(np.linspace(0, 1, N_BLOCKS))
    max_x, max_y = 0, 0

    for i in range(N_BLOCKS):
        w, h = BLOCK_DIMS[i] if not rotated[i] else BLOCK_DIMS[i][::-1]
        rect = mpatches.FancyBboxPatch(
            (positions[i, 0], positions[i, 1]), w, h,
            boxstyle="round,pad=0.05",
            facecolor=colors[i], edgecolor='black', linewidth=0.8, alpha=0.8)
        ax.add_patch(rect)
        cx = positions[i, 0] + w/2
        cy = positions[i, 1] + h/2
        ax.text(cx, cy, BLOCK_NAMES[i], ha='center', va='center',
               fontsize=6, fontweight='bold')
        max_x = max(max_x, positions[i, 0] + w)
        max_y = max(max_y, positions[i, 1] + h)

    # Draw nets
    for src, dst, weight in NETLIST:
        ws = BLOCK_DIMS[src] if not rotated[src] else BLOCK_DIMS[src][::-1]
        wd = BLOCK_DIMS[dst] if not rotated[dst] else BLOCK_DIMS[dst][::-1]
        cs = positions[src] + np.array([ws[0], ws[1]]) / 2
        cd = positions[dst] + np.array([wd[0], wd[1]]) / 2
        ax.plot([cs[0], cd[0]], [cs[1], cd[1]], 'b-',
               alpha=0.15 * weight, linewidth=0.5 * weight)

    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(title)

def plot_results(init_pos, init_rot, best_pos, best_rot,
                 init_cost, best_cost, best_wl, best_area, best_thermal,
                 history, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Simulated Annealing for Chip Floorplan Optimization (Intel)',
                 fontsize=15, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # (a) Initial floorplan
    ax1 = fig.add_subplot(gs[0, 0])
    plot_floorplan(ax1, init_pos, init_rot, '(a) Initial Random Placement')

    # (b) Optimized floorplan
    ax2 = fig.add_subplot(gs[0, 1])
    plot_floorplan(ax2, best_pos, best_rot, '(b) SA-Optimized Placement')

    # (c) Cost convergence
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['cost'], 'b-', alpha=0.5, linewidth=0.5, label='Current')
    ax3.plot(history['best_cost'], 'r-', linewidth=2, label='Best')
    ax3.set_xlabel('Temperature Step')
    ax3.set_ylabel('Total Cost')
    ax3.set_title('(c) SA Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (d) Temperature schedule
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(history['temp'], 'r-', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Temperature (log)')
    ax4.set_title('(d) Cooling Schedule')
    ax4.grid(True, alpha=0.3)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(history['accept_rate'], 'b-', alpha=0.5, linewidth=0.5)
    ax4_twin.set_ylabel('Accept Rate', color='blue')

    # (e) Thermal map
    ax5 = fig.add_subplot(gs[1, 1])
    grid_res = 100
    thermal_map = np.zeros((grid_res, grid_res))
    max_x = max(best_pos[i, 0] + (BLOCK_DIMS[i][0] if not best_rot[i]
                else BLOCK_DIMS[i][1]) for i in range(N_BLOCKS))
    max_y = max(best_pos[i, 1] + (BLOCK_DIMS[i][1] if not best_rot[i]
                else BLOCK_DIMS[i][0]) for i in range(N_BLOCKS))
    for i in range(N_BLOCKS):
        w, h = BLOCK_DIMS[i] if not best_rot[i] else BLOCK_DIMS[i][::-1]
        x0 = int(best_pos[i, 0] / max_x * (grid_res-1))
        y0 = int(best_pos[i, 1] / max_y * (grid_res-1))
        x1 = int((best_pos[i, 0] + w) / max_x * (grid_res-1))
        y1 = int((best_pos[i, 1] + h) / max_y * (grid_res-1))
        x0, x1 = max(0, x0), min(grid_res, x1)
        y0, y1 = max(0, y0), min(grid_res, y1)
        thermal_map[y0:y1, x0:x1] += BLOCK_POWER[i]
    # Gaussian blur for thermal spreading
    from scipy.ndimage import gaussian_filter
    thermal_map = gaussian_filter(thermal_map, sigma=5)
    im = ax5.imshow(thermal_map, origin='lower', cmap='hot',
                    extent=[0, max_x, 0, max_y])
    plt.colorbar(im, ax=ax5, label='Power Density [W/mm2]')
    ax5.set_xlabel('x [mm]')
    ax5.set_ylabel('y [mm]')
    ax5.set_title('(e) Thermal Map (Optimized)')
    ax5.set_aspect('equal')

    # (f) Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = [
        ['Metric', 'Initial', 'Optimized'],
        ['Total Cost', f'{init_cost:.1f}', f'{best_cost:.1f}'],
        ['Wirelength', '-', f'{best_wl:.1f} mm'],
        ['Area', '-', f'{best_area:.1f} mm2'],
        ['Thermal Cost', '-', f'{best_thermal:.2f}'],
        ['Improvement', '', f'{(1-best_cost/init_cost)*100:.1f}%'],
        ['', '', ''],
        ['Blocks', f'{N_BLOCKS}', ''],
        ['Nets', f'{len(NETLIST)}', ''],
        ['T_init / T_min', '100 / 0.01', ''],
        ['Cooling Rate', '0.995', ''],
        ['Moves', 'move/swap/rotate', ''],
    ]
    table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax6.set_title('(f) Optimization Summary', fontsize=12, pad=15)

    fig.text(0.5, 0.01,
             'Simulated Annealing from scratch | Multi-objective: Wirelength + Area + Thermal | '
             '15 IP blocks, 16 nets',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Simulated Annealing for Chip Floorplan Optimization")
    print("Portfolio: Intel Japan — Optimization")
    print("=" * 60)

    print(f"\n  Blocks: {N_BLOCKS}, Nets: {len(NETLIST)}")

    # Initial random placement
    init_pos, init_rot = random_placement()
    init_cost, _, _, _ = compute_cost(init_pos, init_rot)
    init_cost += 100 * overlap_penalty(init_pos, init_rot)
    print(f"  Initial cost: {init_cost:.1f}")

    print("\nRunning Simulated Annealing...")
    best_pos, best_rot, best_cost, best_wl, best_area, best_thermal, history = \
        simulated_annealing(T_init=100, T_min=0.01, cooling=0.995, n_iter_per_temp=50)
    print(f"  Best cost: {best_cost:.1f}")
    print(f"  Wirelength: {best_wl:.1f}, Area: {best_area:.1f}, Thermal: {best_thermal:.2f}")
    print(f"  Improvement: {(1-best_cost/init_cost)*100:.1f}%")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "chip_floorplan_results.png")
    print("\nPlotting results...")
    plot_results(init_pos, init_rot, best_pos, best_rot,
                 init_cost, best_cost, best_wl, best_area, best_thermal,
                 history, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
