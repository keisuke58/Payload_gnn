"""
H3-22S Trajectory Optimization — Step A+B
==========================================
scipy.optimize で pitch profile パラメータと機体パラメータを自動最適化。

Target (Real H3 TF2/3号機):
  - SRB separation: T+116s, ~50 km
  - Fairing jettison: T+207-214s, ~150 km
  - MECO: T+298s, ~200 km
  - SECO: ~T+850-1017s, ~300 km circular orbit

Optimizes:
  - Pitch profile breakpoints and angles
  - S1 propellant mass (real: 222 ton)
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize

# ============================================================
# Constants
# ============================================================
G = 6.674e-11
M_EARTH = 5.972e24
R_EARTH = 6371000.0
MU = G * M_EARTH

TNSC_LAT = 30.4009
V_EARTH_ROT = 2 * math.pi * R_EARTH * math.cos(math.radians(TNSC_LAT)) / 86400

# ============================================================
# Real H3 flight data (targets)
# ============================================================
TARGET_SRB_SEP_TIME = 116.0   # T+116s
TARGET_SRB_SEP_ALT = 50.0     # 50 km
TARGET_MECO_TIME = 298.0       # T+298s
TARGET_MECO_ALT = 200.0        # 200 km
TARGET_FAIRING_TIME = 210.0    # T+207-214s
TARGET_FAIRING_ALT = 150.0     # 150 km (jettison altitude)
TARGET_ORBIT_ALT = 300.0       # 300 km circular


# ============================================================
# Vehicle parameters (fixed during optimization)
# ============================================================
SRB_THRUST = 2 * 1600e3
SRB_PROP_MASS = 2 * 66000
SRB_DRY_MASS = 2 * 11000
SRB_ISP = 280.0

S1_THRUST = 2 * 1471e3
S1_ISP_SL = 380.0
S1_ISP_VAC = 425.0
S1_DRY_MASS = 20000

S2_THRUST = 137e3
S2_ISP = 448.0
S2_PROP_MASS = 28000
S2_DRY_MASS = 4000

FAIRING_MASS = 3400
PAYLOAD_MASS = 4000

COAST_BETWEEN_STAGES = 8.0

# ============================================================
# Parametric pitch profile
# ============================================================
def make_pitch_func(params):
    """Create pitch function from optimization parameters.

    params = [
        p1,   # angle at T=20s (end of kick)
        p2,   # angle at T=60s
        p3,   # angle at T=120s (SRB burnout)
        p4,   # angle at T=200s
        p5,   # angle at T=300s (MECO region)
        p6,   # angle at T=500s (S2 mid-burn)
        p7,   # final angle (>500s)
        s1_prop,  # S1 propellant mass (kg)
    ]
    """
    p1, p2, p3, p4, p5, p6, p7 = params[:7]

    def pitch(t_sec):
        if t_sec < 8:
            return 0.0
        elif t_sec < 20:
            return p1 * (t_sec - 8) / 12
        elif t_sec < 60:
            return p1 + (p2 - p1) * (t_sec - 20) / 40
        elif t_sec < 120:
            return p2 + (p3 - p2) * (t_sec - 60) / 60
        elif t_sec < 200:
            return p3 + (p4 - p3) * (t_sec - 120) / 80
        elif t_sec < 300:
            return p4 + (p5 - p4) * (t_sec - 200) / 100
        elif t_sec < 500:
            return p5 + (p6 - p5) * (t_sec - 300) / 200
        else:
            return p6 + (p7 - p6) * min(1.0, (t_sec - 500) / 300)
    return pitch


def atm_density(h):
    if h < 0:
        return 1.225
    elif h < 11000:
        return 1.225 * math.exp(-h / 8500)
    elif h < 25000:
        return 0.3639 * math.exp(-(h - 11000) / 6340)
    elif h < 50000:
        return 0.04008 * math.exp(-(h - 25000) / 7250)
    elif h < 80000:
        return 1.027e-3 * math.exp(-(h - 50000) / 5800)
    else:
        return 3.097e-5 * math.exp(-(h - 80000) / 5500)


def simulate(params, verbose=False):
    """Run trajectory simulation with given parameters.

    Returns dict with flight events and final state.
    """
    pitch_func = make_pitch_func(params)
    s1_prop_mass = params[7]

    s1_burn_time = s1_prop_mass * S1_ISP_VAC * 9.81 / S1_THRUST  # approximate

    m_liftoff = (SRB_PROP_MASS + SRB_DRY_MASS +
                 s1_prop_mass + S1_DRY_MASS +
                 S2_PROP_MASS + S2_DRY_MASS +
                 FAIRING_MASS + PAYLOAD_MASS)

    dt = 0.5
    r = R_EARTH + 50.0
    theta = 0.0
    v_r = 0.0
    v_theta = V_EARTH_ROT

    m_srb_prop = SRB_PROP_MASS
    m_s1_prop = s1_prop_mass
    m_s2_prop = S2_PROP_MASS
    m_total = m_liftoff
    fairing_jettisoned = False

    phase = "SRB+S1"
    t = 0.0

    result = {
        "srb_sep_time": None, "srb_sep_alt": None,
        "meco_time": None, "meco_alt": None,
        "fairing_time": None, "fairing_alt": None,
        "seco_time": None, "seco_alt": None,
        "orbit_achieved": False,
        "perigee": None, "apogee": None, "eccentricity": None,
        "crashed": False,
        "traj": [] if verbose else None,
    }

    while t < 2000:
        h = r - R_EARTH
        if h < -1000:
            result["crashed"] = True
            break

        g_local = MU / (r * r)
        rho = atm_density(h)

        v_mag = math.sqrt(v_r**2 + v_theta**2)
        drag = 0.5 * rho * v_mag**2 * 0.3 * math.pi * 2.6**2 if v_mag > 0 else 0

        thrust = 0

        if phase == "SRB+S1":
            if m_srb_prop > 0:
                mdot_srb = SRB_THRUST / (SRB_ISP * 9.81)
                dm_srb = min(mdot_srb * dt, m_srb_prop)
                s1_isp = S1_ISP_SL + (S1_ISP_VAC - S1_ISP_SL) * min(1.0, h / 50000)
                mdot_s1 = S1_THRUST / (s1_isp * 9.81)
                dm_s1 = min(mdot_s1 * dt, m_s1_prop)
                thrust = SRB_THRUST + S1_THRUST
                m_srb_prop -= dm_srb
                m_s1_prop -= dm_s1
                m_total -= (dm_srb + dm_s1)
            else:
                m_total -= SRB_DRY_MASS
                phase = "S1_only"
                result["srb_sep_time"] = t
                result["srb_sep_alt"] = h / 1000
                if verbose:
                    print(f"  T+{t:.0f}s: SRB-3 sep — {h/1000:.1f} km, {v_mag:.0f} m/s")
                continue

        elif phase == "S1_only":
            if m_s1_prop > 0:
                s1_isp = S1_ISP_SL + (S1_ISP_VAC - S1_ISP_SL) * min(1.0, h / 50000)
                mdot_s1 = S1_THRUST / (s1_isp * 9.81)
                dm_s1 = min(mdot_s1 * dt, m_s1_prop)
                thrust = S1_THRUST
                m_s1_prop -= dm_s1
                m_total -= dm_s1
            else:
                m_total -= S1_DRY_MASS
                phase = "coast"
                result["meco_time"] = t
                result["meco_alt"] = h / 1000
                coast_start = t
                if verbose:
                    print(f"  T+{t:.0f}s: MECO — {h/1000:.1f} km, {v_mag:.0f} m/s")
                continue

        elif phase == "coast":
            thrust = 0
            if t - coast_start >= COAST_BETWEEN_STAGES:
                phase = "S2"
                continue

        elif phase == "S2":
            v_circ_local = math.sqrt(MU / r)
            if h > 250000 and v_theta >= v_circ_local * 0.997:
                phase = "SECO"
                result["seco_time"] = t
                result["seco_alt"] = h / 1000
                result["orbit_achieved"] = True
                if verbose:
                    print(f"  T+{t:.0f}s: SECO — {h/1000:.1f} km, {v_mag:.0f} m/s")
                # Compute orbital elements
                v_orbital = v_mag
                energy = 0.5 * v_orbital**2 - MU / r
                if energy < 0:
                    a_orbit = -MU / (2 * energy)
                    h_angular = r * v_theta
                    e_orbit = math.sqrt(1 + 2 * energy * h_angular**2 / MU**2)
                    result["perigee"] = (a_orbit * (1 - e_orbit) - R_EARTH) / 1000
                    result["apogee"] = (a_orbit * (1 + e_orbit) - R_EARTH) / 1000
                    result["eccentricity"] = e_orbit
                break
            elif m_s2_prop > 0:
                mdot_s2 = S2_THRUST / (S2_ISP * 9.81)
                dm_s2 = min(mdot_s2 * dt, m_s2_prop)
                thrust = S2_THRUST
                m_s2_prop -= dm_s2
                m_total -= dm_s2
            else:
                result["seco_time"] = t
                result["seco_alt"] = h / 1000
                v_circ_local = math.sqrt(MU / r)
                result["orbit_achieved"] = v_theta >= v_circ_local * 0.90
                break

        elif phase == "SECO":
            break

        # Fairing
        if not fairing_jettisoned and h > 150000:
            m_total -= FAIRING_MASS
            fairing_jettisoned = True
            result["fairing_time"] = t
            result["fairing_alt"] = h / 1000

        pitch = pitch_func(t)
        pitch_rad = math.radians(pitch)

        if thrust > 0:
            thrust_r = thrust * math.cos(pitch_rad)
            thrust_theta = thrust * math.sin(pitch_rad)
        else:
            thrust_r = 0
            thrust_theta = 0

        if v_mag > 0:
            drag_r = -drag * v_r / v_mag
            drag_theta = -drag * v_theta / v_mag
        else:
            drag_r = 0
            drag_theta = 0

        a_r = thrust_r / m_total + drag_r / m_total - g_local + v_theta**2 / r
        a_theta = thrust_theta / m_total + drag_theta / m_total - v_r * v_theta / r

        v_r += a_r * dt
        v_theta += a_theta * dt
        r += v_r * dt
        theta += v_theta / r * dt

        if verbose and result["traj"] is not None:
            result["traj"].append({
                "t": t, "h": h / 1000, "v": v_mag,
                "v_r": v_r, "v_theta": v_theta,
                "downrange": theta * R_EARTH / 1000,
                "phase": phase, "mass": m_total,
                "drag_kN": drag / 1000, "pitch": pitch,
            })

        t += dt

    return result


def objective(params):
    """Cost function: weighted sum of squared errors vs real H3 data."""
    # Monotonicity constraint on pitch angles
    for i in range(6):
        if params[i] > params[i + 1]:
            return 1e6  # penalty

    # Reasonable bounds
    if params[0] < 1 or params[6] > 90:
        return 1e6

    res = simulate(params)

    if res["crashed"]:
        return 1e6

    cost = 0.0

    # SRB separation timing and altitude
    if res["srb_sep_time"] is not None:
        cost += 10.0 * ((res["srb_sep_time"] - TARGET_SRB_SEP_TIME) / TARGET_SRB_SEP_TIME) ** 2
        cost += 5.0 * ((res["srb_sep_alt"] - TARGET_SRB_SEP_ALT) / TARGET_SRB_SEP_ALT) ** 2
    else:
        cost += 100.0

    # MECO timing and altitude (most important)
    if res["meco_time"] is not None:
        cost += 20.0 * ((res["meco_time"] - TARGET_MECO_TIME) / TARGET_MECO_TIME) ** 2
        cost += 15.0 * ((res["meco_alt"] - TARGET_MECO_ALT) / TARGET_MECO_ALT) ** 2
    else:
        cost += 200.0

    # Fairing jettison
    if res["fairing_time"] is not None:
        cost += 3.0 * ((res["fairing_time"] - TARGET_FAIRING_TIME) / TARGET_FAIRING_TIME) ** 2

    # Orbital insertion
    if res["orbit_achieved"]:
        if res["eccentricity"] is not None:
            cost += 10.0 * res["eccentricity"] ** 2  # want circular
            # Prefer orbit near 300 km
            if res["perigee"] is not None:
                cost += 5.0 * ((res["perigee"] - TARGET_ORBIT_ALT) / TARGET_ORBIT_ALT) ** 2
    else:
        cost += 50.0

    return cost


# ============================================================
# Run optimization
# ============================================================
print("=" * 70)
print("  H3-22S Trajectory Optimization")
print("  Target: Match real H3 TF2/3 flight data")
print("=" * 70)

# Initial guess (current hand-tuned values)
x0 = np.array([
    5.0,    # p1: angle at T=20s
    25.0,   # p2: angle at T=60s
    45.0,   # p3: angle at T=120s
    70.0,   # p4: angle at T=200s
    85.0,   # p5: angle at T=300s
    88.0,   # p6: angle at T=500s
    89.0,   # p7: final angle
    222000, # S1 propellant mass (updated to real value)
])

print(f"\n--- Initial guess ---")
res0 = simulate(x0, verbose=True)
cost0 = objective(x0)
print(f"  Cost: {cost0:.4f}")
if res0["orbit_achieved"]:
    print(f"  Orbit: {res0['perigee']:.0f} x {res0['apogee']:.0f} km, e={res0['eccentricity']:.4f}")

# Bounds for optimization
bounds = [
    (2.0, 15.0),     # p1
    (10.0, 40.0),     # p2
    (25.0, 60.0),     # p3
    (45.0, 80.0),     # p4
    (65.0, 89.0),     # p5
    (80.0, 89.5),     # p6
    (85.0, 90.0),     # p7
    (180000, 240000), # S1 propellant mass
]

print(f"\n--- Running Differential Evolution ---")
print(f"  Population: 30, Max generations: 100")

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=100,
    popsize=30,
    tol=1e-6,
    mutation=(0.5, 1.5),
    recombination=0.8,
    disp=True,
    workers=-1,  # parallel
)

print(f"\n--- Optimization Result ---")
print(f"  Success: {result.success}")
print(f"  Cost: {result.fun:.6f}")
print(f"  Iterations: {result.nit}")

opt_params = result.x
print(f"\n  Optimized pitch profile:")
print(f"    T=20s:  {opt_params[0]:.1f}°")
print(f"    T=60s:  {opt_params[1]:.1f}°")
print(f"    T=120s: {opt_params[2]:.1f}°")
print(f"    T=200s: {opt_params[3]:.1f}°")
print(f"    T=300s: {opt_params[4]:.1f}°")
print(f"    T=500s: {opt_params[5]:.1f}°")
print(f"    Final:  {opt_params[6]:.1f}°")
print(f"    S1 propellant: {opt_params[7]/1000:.1f} ton")

# Run final simulation with verbose output
print(f"\n--- Final Simulation ---")
res_final = simulate(opt_params, verbose=True)

print(f"\n=== Comparison: Optimized vs Real H3 ===")
print(f"{'Event':<25} {'Sim':>10} {'Real':>10} {'Error':>10}")
print("-" * 55)

if res_final["srb_sep_time"]:
    err_t = res_final["srb_sep_time"] - TARGET_SRB_SEP_TIME
    err_h = res_final["srb_sep_alt"] - TARGET_SRB_SEP_ALT
    print(f"{'SRB Sep Time (s)':<25} {res_final['srb_sep_time']:>10.1f} {TARGET_SRB_SEP_TIME:>10.1f} {err_t:>+10.1f}")
    print(f"{'SRB Sep Alt (km)':<25} {res_final['srb_sep_alt']:>10.1f} {TARGET_SRB_SEP_ALT:>10.1f} {err_h:>+10.1f}")

if res_final["fairing_time"]:
    err_t = res_final["fairing_time"] - TARGET_FAIRING_TIME
    print(f"{'Fairing Time (s)':<25} {res_final['fairing_time']:>10.1f} {TARGET_FAIRING_TIME:>10.1f} {err_t:>+10.1f}")

if res_final["meco_time"]:
    err_t = res_final["meco_time"] - TARGET_MECO_TIME
    err_h = res_final["meco_alt"] - TARGET_MECO_ALT
    print(f"{'MECO Time (s)':<25} {res_final['meco_time']:>10.1f} {TARGET_MECO_TIME:>10.1f} {err_t:>+10.1f}")
    print(f"{'MECO Alt (km)':<25} {res_final['meco_alt']:>10.1f} {TARGET_MECO_ALT:>10.1f} {err_h:>+10.1f}")

if res_final["seco_time"]:
    print(f"{'SECO Time (s)':<25} {res_final['seco_time']:>10.1f} {'~850-1017':>10} {'':>10}")
    print(f"{'SECO Alt (km)':<25} {res_final['seco_alt']:>10.1f} {'~300':>10} {'':>10}")

if res_final["orbit_achieved"] and res_final["perigee"]:
    print(f"\n=== Orbital Parameters ===")
    print(f"  Perigee:  {res_final['perigee']:.1f} km")
    print(f"  Apogee:   {res_final['apogee']:.1f} km")
    print(f"  Eccentricity: {res_final['eccentricity']:.4f}")

# ============================================================
# Generate updated pitch profile for jaxa_h3_orbital.py
# ============================================================
print(f"\n=== Updated programmed_pitch() for jaxa_h3_orbital.py ===")
print(f"""
def programmed_pitch(t_sec):
    \"\"\"Optimized pitch profile — matches real H3 TF2/3 flight data.
    Auto-tuned by scipy.optimize.differential_evolution.\"\"\"
    if t_sec < 8:
        return 0.0
    elif t_sec < 20:
        return {opt_params[0]:.2f} * (t_sec - 8) / 12
    elif t_sec < 60:
        return {opt_params[0]:.2f} + {opt_params[1] - opt_params[0]:.2f} * (t_sec - 20) / 40
    elif t_sec < 120:
        return {opt_params[1]:.2f} + {opt_params[2] - opt_params[1]:.2f} * (t_sec - 60) / 60
    elif t_sec < 200:
        return {opt_params[2]:.2f} + {opt_params[3] - opt_params[2]:.2f} * (t_sec - 120) / 80
    elif t_sec < 300:
        return {opt_params[3]:.2f} + {opt_params[4] - opt_params[3]:.2f} * (t_sec - 200) / 100
    elif t_sec < 500:
        return {opt_params[4]:.2f} + {opt_params[5] - opt_params[4]:.2f} * (t_sec - 300) / 200
    else:
        return {opt_params[5]:.2f} + {opt_params[6] - opt_params[5]:.2f} * min(1.0, (t_sec - 500) / 300)
""")

# ============================================================
# Convergence + comparison plots
# ============================================================
# Run both old and new for comparison plot
x_old = np.array([5.0, 25.0, 45.0, 70.0, 85.0, 88.0, 89.0, 180000])
res_old = simulate(x_old, verbose=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Pitch profile comparison
ax = axes[0, 0]
ts = np.linspace(0, 800, 500)
pitch_old = make_pitch_func(x_old)
pitch_new = make_pitch_func(opt_params)
ax.plot(ts, [pitch_old(t) for t in ts], "b--", lw=1.5, label="Hand-tuned (before)")
ax.plot(ts, [pitch_new(t) for t in ts], "r-", lw=2, label="Optimized (after)")
ax.axvline(TARGET_SRB_SEP_TIME, color="gray", ls=":", alpha=0.5)
ax.axvline(TARGET_MECO_TIME, color="gray", ls=":", alpha=0.5)
ax.text(TARGET_SRB_SEP_TIME, 5, "SRB Sep", fontsize=7, rotation=90)
ax.text(TARGET_MECO_TIME, 5, "MECO", fontsize=7, rotation=90)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pitch Angle (deg from vertical)")
ax.set_title("(a) Pitch Profile: Before vs After Optimization")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Run detailed trajectory for both
res_old_v = simulate(x_old, verbose=True)
res_new_v = simulate(opt_params, verbose=True)

if res_old_v["traj"] and res_new_v["traj"]:
    t_old = [p["t"] for p in res_old_v["traj"]]
    h_old = [p["h"] for p in res_old_v["traj"]]
    v_old = [p["v"] for p in res_old_v["traj"]]

    t_new = [p["t"] for p in res_new_v["traj"]]
    h_new = [p["h"] for p in res_new_v["traj"]]
    v_new = [p["v"] for p in res_new_v["traj"]]

    # (b) Altitude comparison
    ax = axes[0, 1]
    ax.plot(t_old, h_old, "b--", lw=1.5, label="Before (S1=180t)")
    ax.plot(t_new, h_new, "r-", lw=2, label="Optimized (S1=222t)")
    # Real data points
    ax.plot(TARGET_SRB_SEP_TIME, TARGET_SRB_SEP_ALT, "k^", ms=10, label="Real H3 data")
    ax.plot(TARGET_MECO_TIME, TARGET_MECO_ALT, "k^", ms=10)
    ax.plot(TARGET_FAIRING_TIME, TARGET_FAIRING_ALT, "k^", ms=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("(b) Altitude: Before vs After")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Speed comparison
    ax = axes[1, 0]
    ax.plot(t_old, v_old, "b--", lw=1.5, label="Before")
    ax.plot(t_new, v_new, "r-", lw=2, label="Optimized")
    ax.axhline(y=7800, color="green", ls="--", alpha=0.5, label="Orbital velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("(c) Speed: Before vs After")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# (d) Error bar chart
ax = axes[1, 1]
labels = []
errors_old = []
errors_new = []

if res_old_v["srb_sep_time"] and res_new_v["srb_sep_time"]:
    labels.append("SRB Sep\nTime")
    errors_old.append(abs(res_old_v["srb_sep_time"] - TARGET_SRB_SEP_TIME) / TARGET_SRB_SEP_TIME * 100)
    errors_new.append(abs(res_new_v["srb_sep_time"] - TARGET_SRB_SEP_TIME) / TARGET_SRB_SEP_TIME * 100)

    labels.append("SRB Sep\nAlt")
    errors_old.append(abs(res_old_v["srb_sep_alt"] - TARGET_SRB_SEP_ALT) / TARGET_SRB_SEP_ALT * 100)
    errors_new.append(abs(res_new_v["srb_sep_alt"] - TARGET_SRB_SEP_ALT) / TARGET_SRB_SEP_ALT * 100)

if res_old_v["meco_time"] and res_new_v["meco_time"]:
    labels.append("MECO\nTime")
    errors_old.append(abs(res_old_v["meco_time"] - TARGET_MECO_TIME) / TARGET_MECO_TIME * 100)
    errors_new.append(abs(res_new_v["meco_time"] - TARGET_MECO_TIME) / TARGET_MECO_TIME * 100)

    labels.append("MECO\nAlt")
    errors_old.append(abs(res_old_v["meco_alt"] - TARGET_MECO_ALT) / TARGET_MECO_ALT * 100)
    errors_new.append(abs(res_new_v["meco_alt"] - TARGET_MECO_ALT) / TARGET_MECO_ALT * 100)

x_pos = np.arange(len(labels))
ax.bar(x_pos - 0.2, errors_old, 0.35, label="Before", color="steelblue", alpha=0.7)
ax.bar(x_pos + 0.2, errors_new, 0.35, label="Optimized", color="firebrick", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Error (%)")
ax.set_title("(d) Error vs Real H3 Data")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("H3-22S Trajectory Optimization — scipy.optimize", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("h3_optimization_results.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: h3_optimization_results.png")
