#!/usr/bin/env python3
"""
NSGA-II Multi-Objective Process Optimization
==============================================
Portfolio: TSMC / JASM — ML/Optimization

Optimizes semiconductor fabrication process parameters
for multiple conflicting objectives using NSGA-II.

Optimization Concepts:
  - NSGA-II from scratch (non-dominated sorting + crowding distance)
  - Pareto front visualization
  - Multi-objective: yield vs throughput vs defect density
  - Real-coded genetic operators (SBX crossover, polynomial mutation)

Domain:
  - Advanced node process window optimization
  - Litho-etch-deposition parameter tuning
  - Yield-throughput-quality tradeoff

Author: Nishioka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

np.random.seed(42)

# ============================================================
# Process Objectives (3-objective problem)
# ============================================================
PARAM_NAMES = ['Exposure Dose', 'Focus Offset', 'Etch Time',
               'Dep Temp', 'CMP Speed', 'Anneal Time']
BOUNDS = np.array([
    [10, 50],     # mJ/cm2
    [-0.3, 0.3],  # um
    [30, 120],    # s
    [300, 600],   # C
    [50, 200],    # rpm
    [10, 120],    # s
])

def evaluate(X):
    """3-objective process evaluation.
    Minimize: f1 = -yield, f2 = -throughput, f3 = defect_density
    """
    n = len(X)
    x_norm = (X - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])

    # f1: Yield (maximize → minimize negative)
    yield_val = 95 - 10*(x_norm[:, 1]**2) - 5*(x_norm[:, 0]-0.6)**2 \
                - 3*(x_norm[:, 3]-0.5)**2 - 2*x_norm[:, 2]*x_norm[:, 4]
    yield_val += np.random.randn(n) * 0.5

    # f2: Throughput (maximize → minimize negative)
    throughput = 100 / (x_norm[:, 2]*90+30 + x_norm[:, 5]*110+10) * 3600
    throughput *= (1 + 0.3*x_norm[:, 4])

    # f3: Defect density (minimize)
    defects = 0.5 + 2*np.abs(x_norm[:, 1]-0.5) + 1.5*(1-x_norm[:, 0]) \
              + x_norm[:, 4]**2 + 0.5*np.random.rand(n)

    return np.column_stack([-yield_val, -throughput, defects])

# ============================================================
# NSGA-II (from scratch)
# ============================================================
def non_dominated_sort(objectives):
    """Fast non-dominated sorting (NSGA-II)."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    ranks = np.zeros(n, dtype=int)
    fronts = [[]]

    for i in range(n):
        for j in range(i+1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return ranks, fronts[:-1]

def dominates(a, b):
    """Check if solution a dominates b (all objectives minimized)."""
    return np.all(a <= b) and np.any(a < b)

def crowding_distance(objectives, front):
    """Compute crowding distance for a front."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    n_obj = objectives.shape[1]

    for m in range(n_obj):
        sorted_idx = np.argsort(objectives[front, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = objectives[front[sorted_idx[-1]], m] - \
                    objectives[front[sorted_idx[0]], m]
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += \
                (objectives[front[sorted_idx[i+1]], m] -
                 objectives[front[sorted_idx[i-1]], m]) / obj_range

    return distances

def sbx_crossover(p1, p2, bounds, eta=20):
    """Simulated Binary Crossover (SBX)."""
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    for i in range(n):
        if np.random.rand() < 0.5:
            if abs(p1[i] - p2[i]) > 1e-10:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1[i] = np.clip(c1[i], bounds[i, 0], bounds[i, 1])
        c2[i] = np.clip(c2[i], bounds[i, 0], bounds[i, 1])
    return c1, c2

def polynomial_mutation(x, bounds, eta=20, prob=None):
    """Polynomial mutation."""
    n = len(x)
    if prob is None:
        prob = 1.0 / n
    child = x.copy()
    for i in range(n):
        if np.random.rand() < prob:
            delta = (x[i] - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0])
            u = np.random.rand()
            if u < 0.5:
                deltaq = (2*u + (1-2*u)*(1-delta)**(eta+1))**(1/(eta+1)) - 1
            else:
                deltaq = 1 - (2*(1-u) + 2*(u-0.5)*delta**(eta+1))**(1/(eta+1))
            child[i] = x[i] + deltaq * (bounds[i, 1] - bounds[i, 0])
            child[i] = np.clip(child[i], bounds[i, 0], bounds[i, 1])
    return child

def tournament_selection(ranks, crowding, pop_size):
    """Binary tournament selection based on rank and crowding distance."""
    selected = []
    for _ in range(pop_size):
        i, j = np.random.randint(0, pop_size, 2)
        if ranks[i] < ranks[j]:
            selected.append(i)
        elif ranks[i] > ranks[j]:
            selected.append(j)
        elif crowding[i] > crowding[j]:
            selected.append(i)
        else:
            selected.append(j)
    return selected

def nsga2(pop_size=100, n_gen=80):
    """Run NSGA-II optimization."""
    n_var = len(BOUNDS)

    # Initialize population
    pop = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (pop_size, n_var))
    obj = evaluate(pop)

    history = {'hv': [], 'n_pareto': []}

    for gen in range(n_gen):
        # Create offspring
        ranks, fronts = non_dominated_sort(obj)

        # Compute crowding for all
        crowding = np.zeros(pop_size)
        for front in fronts:
            if len(front) > 0:
                cd = crowding_distance(obj, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]

        # Tournament selection
        parents = tournament_selection(ranks, crowding, pop_size)

        # Crossover + mutation
        offspring = np.zeros((pop_size, n_var))
        for i in range(0, pop_size, 2):
            p1, p2 = pop[parents[i]], pop[parents[min(i+1, pop_size-1)]]
            c1, c2 = sbx_crossover(p1, p2, BOUNDS)
            offspring[i] = polynomial_mutation(c1, BOUNDS)
            if i + 1 < pop_size:
                offspring[i+1] = polynomial_mutation(c2, BOUNDS)

        off_obj = evaluate(offspring)

        # Combine parent + offspring
        combined_pop = np.vstack([pop, offspring])
        combined_obj = np.vstack([obj, off_obj])

        # Non-dominated sorting on combined
        ranks_c, fronts_c = non_dominated_sort(combined_obj)

        # Select next generation
        new_pop = np.zeros((pop_size, n_var))
        new_obj = np.zeros((pop_size, 3))
        idx = 0
        for front in fronts_c:
            if idx + len(front) <= pop_size:
                for j in front:
                    new_pop[idx] = combined_pop[j]
                    new_obj[idx] = combined_obj[j]
                    idx += 1
            else:
                # Last front: sort by crowding distance
                cd = crowding_distance(combined_obj, front)
                sorted_cd = np.argsort(-cd)
                for k in range(pop_size - idx):
                    j = front[sorted_cd[k]]
                    new_pop[idx] = combined_pop[j]
                    new_obj[idx] = combined_obj[j]
                    idx += 1
                break

        pop = new_pop
        obj = new_obj

        # Track metrics
        pareto = [i for i in range(pop_size) if ranks_c[i] == 0
                  and i < pop_size]
        ranks_new, _ = non_dominated_sort(obj)
        n_pareto = np.sum(ranks_new == 0)
        history['n_pareto'].append(n_pareto)

        if (gen + 1) % 20 == 0:
            print(f"  Gen {gen+1}: Pareto front size = {n_pareto}")

    return pop, obj, history

# ============================================================
# Visualization
# ============================================================
def plot_results(pop, obj, history, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('NSGA-II Multi-Objective Process Optimization (TSMC/JASM)',
                 fontsize=15, fontweight='bold')

    # Get Pareto front
    ranks, fronts = non_dominated_sort(obj)
    pareto_mask = ranks == 0
    pareto_obj = obj[pareto_mask]

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # (a) 3D Pareto front
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.scatter(-obj[~pareto_mask, 0], -obj[~pareto_mask, 1],
               obj[~pareto_mask, 2], s=10, alpha=0.2, c='gray')
    ax1.scatter(-pareto_obj[:, 0], -pareto_obj[:, 1],
               pareto_obj[:, 2], s=40, alpha=0.8, c='red')
    ax1.set_xlabel('Yield [%]', fontsize=8)
    ax1.set_ylabel('Throughput [wph]', fontsize=8)
    ax1.set_zlabel('Defects [/cm2]', fontsize=8)
    ax1.set_title('(a) 3D Pareto Front')

    # (b) Yield vs Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(-obj[~pareto_mask, 0], -obj[~pareto_mask, 1],
               s=10, alpha=0.2, c='gray', label='Dominated')
    ax2.scatter(-pareto_obj[:, 0], -pareto_obj[:, 1],
               s=40, alpha=0.8, c='red', label='Pareto')
    ax2.set_xlabel('Yield [%]')
    ax2.set_ylabel('Throughput [wph]')
    ax2.set_title('(b) Yield vs Throughput')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (c) Yield vs Defects
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(-obj[:, 0], obj[:, 2], c=-obj[:, 1],
                    cmap='viridis', s=20, alpha=0.6)
    ax3.scatter(-pareto_obj[:, 0], pareto_obj[:, 2],
               s=60, facecolors='none', edgecolors='red', linewidths=2)
    plt.colorbar(sc, ax=ax3, label='Throughput [wph]')
    ax3.set_xlabel('Yield [%]')
    ax3.set_ylabel('Defect Density [/cm2]')
    ax3.set_title('(c) Yield vs Defects (color=Throughput)')
    ax3.grid(True, alpha=0.3)

    # (d) Pareto front size evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['n_pareto'], 'b-', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Pareto Front Size')
    ax4.set_title('(d) Pareto Front Evolution')
    ax4.grid(True, alpha=0.3)

    # (e) Parameter parallel coordinates (Pareto solutions)
    ax5 = fig.add_subplot(gs[1, 1])
    pareto_pop = pop[pareto_mask]
    pareto_norm = (pareto_pop - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    for i in range(len(pareto_norm)):
        alpha = 0.3 + 0.7 * (-pareto_obj[i, 0] - (-pareto_obj[:, 0]).min()) / \
                ((-pareto_obj[:, 0]).max() - (-pareto_obj[:, 0]).min() + 1e-10)
        ax5.plot(range(6), pareto_norm[i], 'b-', alpha=alpha, linewidth=0.8)
    ax5.set_xticks(range(6))
    ax5.set_xticklabels(PARAM_NAMES, fontsize=8, rotation=30)
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('(e) Pareto Solutions (Parallel Coordinates)')
    ax5.grid(True, alpha=0.3)

    # (f) Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    best_yield_idx = np.argmax(-pareto_obj[:, 0])
    best_thru_idx = np.argmax(-pareto_obj[:, 1])
    best_defect_idx = np.argmin(pareto_obj[:, 2])
    summary = [
        ['Metric', 'Best Yield', 'Best Throughput', 'Lowest Defect'],
        ['Yield [%]', f'{-pareto_obj[best_yield_idx, 0]:.1f}',
         f'{-pareto_obj[best_thru_idx, 0]:.1f}',
         f'{-pareto_obj[best_defect_idx, 0]:.1f}'],
        ['Throughput', f'{-pareto_obj[best_yield_idx, 1]:.1f}',
         f'{-pareto_obj[best_thru_idx, 1]:.1f}',
         f'{-pareto_obj[best_defect_idx, 1]:.1f}'],
        ['Defects', f'{pareto_obj[best_yield_idx, 2]:.2f}',
         f'{pareto_obj[best_thru_idx, 2]:.2f}',
         f'{pareto_obj[best_defect_idx, 2]:.2f}'],
        ['', '', '', ''],
        ['Pop Size', '100', '', ''],
        ['Generations', '80', '', ''],
        ['Pareto Size', f'{pareto_mask.sum()}', '', ''],
        ['Crossover', 'SBX (eta=20)', '', ''],
        ['Mutation', 'Polynomial', '', ''],
    ]
    table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                       cellLoc='center', loc='center',
                       colColours=['#e6f3ff']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    ax6.set_title('(f) Optimization Summary', fontsize=12, pad=15)

    fig.text(0.5, 0.01,
             'NSGA-II from scratch | Non-dominated sorting + Crowding distance | '
             'SBX crossover | 3-objective process optimization',
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
    print("NSGA-II Multi-Objective Process Optimization")
    print("Portfolio: TSMC / JASM — Optimization")
    print("=" * 60)

    print("\nRunning NSGA-II (pop=100, gen=80)...")
    pop, obj, history = nsga2(pop_size=100, n_gen=80)

    ranks, _ = non_dominated_sort(obj)
    n_pareto = np.sum(ranks == 0)
    pareto_obj = obj[ranks == 0]
    print(f"\n  Pareto front size: {n_pareto}")
    print(f"  Yield range: {-pareto_obj[:,0].max():.1f} - {-pareto_obj[:,0].min():.1f}%")
    print(f"  Throughput range: {-pareto_obj[:,1].max():.1f} - {-pareto_obj[:,1].min():.1f} wph")
    print(f"  Defect range: {pareto_obj[:,2].min():.2f} - {pareto_obj[:,2].max():.2f} /cm2")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "process_optimization_results.png")
    print("\nPlotting results...")
    plot_results(pop, obj, history, save_path)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
