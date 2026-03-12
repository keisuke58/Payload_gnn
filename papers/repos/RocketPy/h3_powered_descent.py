"""
H3 Booster Powered Descent Guidance — Convex Optimization (G-FOLD)
==================================================================
Step C: Fuel-optimal powered descent via SOCP.

Scenario: H3 1st stage booster hoverslam landing
  - LE-9 T/W > 2 at min throttle → cannot hover → hoverslam required
  - Formulation allows coast phases (engine off) + powered braking
  - Convex relaxation of thrust magnitude constraint

Reference:
  - Acikmese & Ploen (2007): Convex PDG for Mars Landing
  - Blackmore et al. (2013): Lossless Convexification (IEEE TCST)
"""
import math
import sys

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# H3 1st Stage Parameters
# ============================================================
M_DRY = 20000.0          # 20 ton dry
M_FUEL = 5000.0           # 5 ton landing fuel
M0 = M_DRY + M_FUEL

THRUST_MAX = 1471e3       # single LE-9
THRUST_MIN = 0.0          # can shut off (hoverslam)
ISP = 400.0
G0 = 9.81
ALPHA = 1.0 / (ISP * G0)

# Initial state (after boostback + re-entry + aero braking)
R0 = np.array([1500.0, 300.0, -100.0])  # [alt, east, north] m
V0 = np.array([-60.0, 10.0, -5.0])      # m/s, descending

GRAVITY = np.array([-G0, 0.0, 0.0])
GLIDE_SLOPE_DEG = 10.0
V_MAX = 90.0

TF = 22.0   # landing time
N = 50
DT = TF / N

print("=" * 70)
print("  H3 Booster Powered Descent — G-FOLD (Hoverslam)")
print("=" * 70)
print(f"  Mass: {M0/1000:.0f} ton, Thrust: 0-{THRUST_MAX/1e3:.0f} kN")
print(f"  T/W at max: {THRUST_MAX/(M0*G0):.1f}")
print(f"  Initial: alt {R0[0]:.0f}m, v_down={-V0[0]:.0f} m/s")


def solve_pdg(tf, n_steps=50):
    """Solve powered descent guidance for given final time."""
    dt = tf / n_steps

    # Variables
    r = cp.Variable((3, n_steps))
    v = cp.Variable((3, n_steps))
    T = cp.Variable((3, n_steps))   # thrust force vector (N)
    sigma = cp.Variable(n_steps)     # thrust magnitude slack

    cons = []

    # Boundary conditions
    cons += [
        r[:, 0] == R0,
        v[:, 0] == V0,
        r[0, -1] == 0.0,
        r[1, -1] == 0.0,
        r[2, -1] == 0.0,
        v[:, -1] == np.zeros(3),
    ]

    # Mass approximation: use initial mass (small fuel fraction)
    # This simplification works when fuel << total mass
    m_avg = M0 - M_FUEL * 0.5  # average mass

    # Dynamics
    for k in range(n_steps - 1):
        cons += [
            v[:, k+1] == v[:, k] + dt * (T[:, k] / m_avg + GRAVITY),
            r[:, k+1] == r[:, k] + dt * 0.5 * (v[:, k] + v[:, k+1]),
        ]

    for k in range(n_steps):
        # SOC constraint: ||T|| <= sigma
        cons += [cp.norm(T[:, k]) <= sigma[k]]

        # Thrust bounds
        cons += [
            sigma[k] >= 0,
            sigma[k] <= THRUST_MAX,
        ]

        # Pointing: thrust mostly upward when on
        cons += [T[0, k] >= 0]  # upward component non-negative

        # Altitude >= 0 during descent
        cons += [r[0, k] >= 0]

        # Velocity limit
        cons += [cp.norm(v[:, k]) <= V_MAX]

        # Glide slope
        if k < n_steps - 1:
            gamma_gs = math.tan(math.radians(GLIDE_SLOPE_DEG))
            cons += [r[0, k] >= gamma_gs * cp.norm(r[1:, k])]

    # Objective: minimize fuel (proportional to integral of thrust magnitude)
    obj = cp.Minimize(cp.sum(sigma) * dt * ALPHA)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    return prob, r, v, T, sigma, dt


# Search over landing time
print(f"\n--- Searching optimal landing time ---")
best = None
results = []

for tf_try in np.arange(15, 35, 1.0):
    prob, r, v, T, sigma, dt = solve_pdg(tf_try, N)
    fuel = prob.value if prob.status in ["optimal", "optimal_inaccurate"] else None
    if fuel is not None:
        results.append((tf_try, fuel, r.value, v.value, T.value, sigma.value, dt))
        if best is None or fuel < best[1]:
            best = (tf_try, fuel, r.value, v.value, T.value, sigma.value, dt)
        print(f"  tf={tf_try:.0f}s: fuel={fuel:.0f} kg ✓")
    else:
        print(f"  tf={tf_try:.0f}s: {prob.status}")

if best is None:
    print("ERROR: No feasible solution found.")
    sys.exit(1)

TF_OPT, fuel_frac, r_val, v_val, T_val, sigma_val, DT_OPT = best
times = np.linspace(0, TF_OPT, N)

# Fuel = integral of mdot dt = integral of (T / (Isp*g0)) dt = sum(sigma) * dt * alpha
fuel_kg = np.sum(sigma_val) * DT_OPT * ALPHA
m_final = M0 - fuel_kg

r_final = r_val[:, -1]
v_final = v_val[:, -1]

print(f"\n=== Optimal Powered Descent ===")
print(f"  Landing time:   {TF_OPT:.1f} s")
print(f"  Final position: ({r_final[0]:.2f}, {r_final[1]:.2f}, {r_final[2]:.2f}) m")
print(f"  Final speed:    {np.linalg.norm(v_final):.4f} m/s")
print(f"  Fuel used:      {fuel_kg:.0f} kg ({fuel_kg/M_FUEL*100:.1f}% of budget)")
print(f"  Max thrust:     {np.max(sigma_val)/1e3:.0f} kN ({np.max(sigma_val)/THRUST_MAX*100:.0f}%)")

# ============================================================
# Plots
# ============================================================
fig = plt.figure(figsize=(18, 12))

# (a) 3D Trajectory
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot(r_val[1, :], r_val[2, :], r_val[0, :], 'b-', lw=2)
ax.scatter(R0[1], R0[2], R0[0], c='red', s=100, marker='^', label='Entry', zorder=5)
ax.scatter(0, 0, 0, c='green', s=150, marker='*', label='Landing', zorder=5)

skip = max(1, N // 12)
scale_f = 0.05 * R0[0]
for k in range(0, N, skip):
    t_norm = np.linalg.norm(T_val[:, k])
    if t_norm > 1000:
        t_dir = T_val[:, k] / t_norm * scale_f * (t_norm / THRUST_MAX)
        ax.quiver(r_val[1, k], r_val[2, k], r_val[0, k],
                  t_dir[1], t_dir[2], t_dir[0],
                  color='orange', alpha=0.7, arrow_length_ratio=0.15)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('(a) 3D Trajectory + Thrust Vectors')
ax.legend(fontsize=8)

# (b) Altitude vs Time
ax = fig.add_subplot(2, 3, 2)
ax.plot(times, r_val[0, :], 'b-', lw=2)
ax.fill_between(times, 0, r_val[0, :], alpha=0.1, color='blue')
ax.axhline(y=0, color='green', ls='-', alpha=0.8, lw=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Altitude (m)')
ax.set_title('(b) Altitude — Hoverslam Profile')
ax.grid(True, alpha=0.3)

# (c) Speed components
ax = fig.add_subplot(2, 3, 3)
speed = np.sqrt(np.sum(v_val**2, axis=0))
ax.plot(times, speed, 'k-', lw=2.5, label='|v|')
ax.plot(times, -v_val[0, :], 'b--', lw=1.5, label='v_down', alpha=0.7)
ax.plot(times, np.sqrt(v_val[1, :]**2 + v_val[2, :]**2), 'r--', lw=1.5, label='v_horiz', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('(c) Velocity Components')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Thrust profile (the key plot — shows bang-off-bang structure)
ax = fig.add_subplot(2, 3, 4)
ax.plot(times, sigma_val / 1e3, 'orange', lw=2.5, label='Optimal thrust')
ax.axhline(y=THRUST_MAX / 1e3, color='red', ls='--', alpha=0.5, label=f'Max {THRUST_MAX/1e3:.0f} kN')
ax.fill_between(times, 0, sigma_val / 1e3, alpha=0.2, color='orange')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thrust (kN)')
ax.set_title('(d) Fuel-Optimal Thrust — Hoverslam')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (e) Thrust components
ax = fig.add_subplot(2, 3, 5)
ax.plot(times, T_val[0, :] / 1e3, 'b-', lw=2, label='T_up')
ax.plot(times, T_val[1, :] / 1e3, 'r-', lw=1.5, label='T_east', alpha=0.7)
ax.plot(times, T_val[2, :] / 1e3, 'g-', lw=1.5, label='T_north', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thrust Component (kN)')
ax.set_title('(e) Thrust Vector Components')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (f) Approach profile with glide slope
ax = fig.add_subplot(2, 3, 6)
horiz = np.sqrt(r_val[1, :]**2 + r_val[2, :]**2)
ax.plot(horiz, r_val[0, :], 'b-', lw=2, label='Trajectory')
h_range = np.linspace(0, max(horiz) * 1.3, 50)
gamma_gs = math.tan(math.radians(GLIDE_SLOPE_DEG))
ax.plot(h_range, gamma_gs * h_range, 'r--', alpha=0.5, label=f'Glide slope {GLIDE_SLOPE_DEG}°')
ax.scatter(horiz[0], r_val[0, 0], c='red', s=80, marker='^', zorder=5)
ax.scatter(0, 0, c='green', s=80, marker='*', zorder=5)
# Color by thrust level
for k in range(N - 1):
    thrust_frac = sigma_val[k] / THRUST_MAX
    color = plt.cm.YlOrRd(thrust_frac * 0.8 + 0.2) if thrust_frac > 0.01 else 'lightblue'
    ax.plot(horiz[k:k+2], r_val[0, k:k+2], '-', color=color, lw=3, alpha=0.8)
ax.set_xlabel('Horizontal Distance (m)')
ax.set_ylabel('Altitude (m)')
ax.set_title('(f) Approach Profile (color=thrust)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"H3 Booster Powered Descent — G-FOLD Convex Optimization (Hoverslam)\n"
    f"Fuel: {fuel_kg:.0f} kg / {M_FUEL:.0f} kg  |  "
    f"Landing speed: {np.linalg.norm(v_final):.3f} m/s  |  tf={TF_OPT:.1f}s",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig("h3_powered_descent.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: h3_powered_descent.png")

# Fuel optimality summary
print(f"\n=== Fuel Optimality Analysis ===")
# Tsiolkovsky: delta-v needed
dv_needed = np.linalg.norm(V0) + math.sqrt(2 * G0 * R0[0])  # rough
fuel_tsiolkovsky = M0 * (1 - math.exp(-dv_needed / (ISP * G0)))
print(f"  delta-v needed (approx): {dv_needed:.0f} m/s")
print(f"  Tsiolkovsky ideal fuel:  {fuel_tsiolkovsky:.0f} kg")
print(f"  G-FOLD optimal fuel:     {fuel_kg:.0f} kg")
print(f"  Efficiency: {fuel_tsiolkovsky/fuel_kg*100:.0f}% of theoretical minimum")
