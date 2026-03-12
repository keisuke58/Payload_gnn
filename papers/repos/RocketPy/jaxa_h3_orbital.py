"""
JAXA H3 Rocket — Full Launch-to-Orbit Simulation
=================================================
Multi-stage launch vehicle trajectory from Tanegashima to LEO/GTO.

Physics model:
  - Gravity turn trajectory with realistic staging
  - Atmospheric drag model (exponential)
  - Earth curvature and rotation
  - Multi-stage: SRB-3 + 1st stage (LE-9) + 2nd stage (LE-5B-3)

H3-22S Configuration:
  - 2x SRB-3 boosters (2,158 kN each, 110s burn)
  - 1st stage: 2x LE-9 (1,471 kN each, LOX/LH2)
  - 2nd stage: 1x LE-5B-3 (137 kN, LOX/LH2)
  - Total liftoff mass: ~420,000 kg
  - Target orbit: 300 km circular LEO (or GTO)

Outputs KML for Google Earth with full trajectory.
"""
import datetime
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Constants
# ============================================================
G = 6.674e-11
M_EARTH = 5.972e24
R_EARTH = 6371000.0
MU = G * M_EARTH  # 3.986e14

# Launch site: Tanegashima Space Center, Yoshinobu LP-2
TNSC_LAT = 30.4009
TNSC_LON = 130.9751
TNSC_ELEV = 50.0

# Earth rotation speed at launch latitude
V_EARTH_ROT = 2 * math.pi * R_EARTH * math.cos(math.radians(TNSC_LAT)) / 86400  # ~400 m/s

# Launch azimuth (eastward for LEO, ~90°)
AZIMUTH_DEG = 90.0
AZIMUTH = math.radians(AZIMUTH_DEG)

print("=" * 70)
print("  JAXA H3-22S ロケット — 軌道投入シミュレーション")
print("  種子島宇宙センター 吉信射点 → LEO 300km")
print("=" * 70)

# ============================================================
# H3-22S Vehicle Parameters
# ============================================================
# SRB-3 (x2) — average thrust (max 2,158 kN each, avg ~1,600 kN)
SRB_THRUST = 2 * 1600e3      # average total 3,200 kN
SRB_BURN_TIME = 113.0         # 113 s
SRB_PROP_MASS = 2 * 66000     # 66 ton propellant each (JAXA spec)
SRB_DRY_MASS = 2 * 11000      # ~11 ton dry each
SRB_ISP = 280.0               # solid motor sea-level Isp

# 1st stage (LE-9 x2)
S1_THRUST = 2 * 1471e3        # 2x 1,471 kN = 2,942 kN
S1_BURN_TIME = 298.0          # ~5 min total (T+0 to T+298)
S1_ISP_SL = 380.0             # sea level
S1_ISP_VAC = 425.0            # vacuum
S1_PROP_MASS = 209400          # LOX/LH2 — optimized (real ~222t, SpaceflightNow)
S1_DRY_MASS = 20000

# 2nd stage (LE-5B-3)
S2_THRUST = 137e3             # 137 kN
S2_BURN_TIME = 534.0          # ~9 min
S2_ISP = 448.0                # vacuum Isp
S2_PROP_MASS = 28000           # LOX/LH2
S2_DRY_MASS = 4000

# Fairing
FAIRING_MASS = 3400            # ~3.4 ton (long fairing)
FAIRING_JETTISON_ALT = 150000  # 150 km

# Payload
PAYLOAD_MASS = 4000            # 4 ton to LEO

# Total liftoff mass
M_LIFTOFF = (SRB_PROP_MASS + SRB_DRY_MASS +
             S1_PROP_MASS + S1_DRY_MASS +
             S2_PROP_MASS + S2_DRY_MASS +
             FAIRING_MASS + PAYLOAD_MASS)

print(f"\n=== H3-22S Configuration ===")
print(f"  Liftoff mass: {M_LIFTOFF/1000:.0f} ton")
print(f"  SRB-3 (x2): {SRB_THRUST/1e6:.1f} MN, {SRB_BURN_TIME:.0f}s burn")
print(f"  1st stage (LE-9 x2): {S1_THRUST/1e6:.2f} MN, Isp {S1_ISP_VAC:.0f}s")
print(f"  2nd stage (LE-5B-3): {S2_THRUST/1e3:.0f} kN, Isp {S2_ISP:.0f}s")
print(f"  Payload: {PAYLOAD_MASS/1000:.0f} ton to LEO")
print(f"  Total liftoff thrust: {(SRB_THRUST + S1_THRUST)/1e6:.1f} MN")
print(f"  T/W ratio: {(SRB_THRUST + S1_THRUST)/(M_LIFTOFF * 9.81):.2f}")

# ============================================================
# Atmospheric model
# ============================================================
def atm_density(h):
    """Exponential atmosphere model."""
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

def gravity(r):
    """Gravity at distance r from Earth center."""
    return MU / (r * r)

# ============================================================
# Simulation: 2D gravity turn in polar coordinates
# ============================================================
dt = 0.5  # time step

# State: [r, theta, v_r, v_theta]
# r = distance from Earth center
# theta = angle from launch site (along surface)
# v_r = radial velocity
# v_theta = tangential velocity

r = R_EARTH + TNSC_ELEV
theta = 0.0
v_r = 0.0
v_theta = V_EARTH_ROT  # Earth rotation

# Mass tracking
m_srb_prop = SRB_PROP_MASS
m_s1_prop = S1_PROP_MASS
m_s2_prop = S2_PROP_MASS
m_total = M_LIFTOFF
fairing_jettisoned = False

# Staging flags
srb_separated = False
s1_separated = False
orbit_achieved = False

# Trajectory storage
traj = []
events = []
t = 0.0

# Programmed pitch profile (degrees from vertical)
# Real rockets use a pre-programmed pitch-over maneuver
def programmed_pitch(t_sec):
    """Optimized pitch profile — matches real H3 TF2/3 flight data.
    Auto-tuned by scipy.optimize.differential_evolution.
    SRB sep T+114s/47km, MECO T+288s/203km, SECO T+766s/363km."""
    if t_sec < 8:
        return 0.0
    elif t_sec < 20:
        return 4.29 * (t_sec - 8) / 12
    elif t_sec < 60:
        return 4.29 + 5.72 * (t_sec - 20) / 40
    elif t_sec < 120:
        return 10.02 + 44.37 * (t_sec - 60) / 60
    elif t_sec < 200:
        return 54.38 + 22.67 * (t_sec - 120) / 80
    elif t_sec < 300:
        return 77.06 + 0.10 * (t_sec - 200) / 100
    elif t_sec < 500:
        return 77.16 + 2.93 * (t_sec - 300) / 200
    else:
        return 80.08 + 9.79 * min(1.0, (t_sec - 500) / 300)

COAST_BETWEEN_STAGES = 8.0  # 8s coast between 1st and 2nd stage

# Track current flight phase
phase = "SRB+S1"  # SRB+S1, S1_only, coast, S2

# Pitch angle (from vertical): 0 = straight up, 90 = horizontal
pitch = 0.0

print(f"\n=== Running Trajectory Simulation ===")

debug_interval = 50  # print every 50s
next_debug = debug_interval

while t < 2000 and r > R_EARTH - 1000:
    h = r - R_EARTH  # altitude
    g_local = gravity(r)
    rho = atm_density(h)

    # Drag
    v_mag = math.sqrt(v_r**2 + v_theta**2)
    Cd = 0.3
    A_ref = math.pi * 2.6**2  # 5.2m diameter → radius 2.6m
    drag = 0.5 * rho * v_mag**2 * Cd * A_ref if v_mag > 0 else 0

    # --- Staging logic ---
    thrust = 0
    mdot = 0

    if phase == "SRB+S1":
        # Both SRB and S1 firing
        if m_srb_prop > 0:
            isp_srb = SRB_ISP
            mdot_srb = SRB_THRUST / (isp_srb * 9.81)
            dm_srb = min(mdot_srb * dt, m_srb_prop)

            # S1 Isp varies with altitude
            s1_isp = S1_ISP_SL + (S1_ISP_VAC - S1_ISP_SL) * min(1.0, h / 50000)
            mdot_s1 = S1_THRUST / (s1_isp * 9.81)
            dm_s1 = min(mdot_s1 * dt, m_s1_prop)

            thrust = SRB_THRUST + S1_THRUST
            mdot = (dm_srb + dm_s1) / dt
            m_srb_prop -= dm_srb
            m_s1_prop -= dm_s1
            m_total -= (dm_srb + dm_s1)
        else:
            # SRB burnout → separate
            m_total -= SRB_DRY_MASS
            phase = "S1_only"
            srb_separated = True
            events.append(("SRB-3 分離", t, h/1000, v_mag, theta * R_EARTH / 1000))
            print(f"  T+{t:.0f}s: SRB-3 separation — Alt {h/1000:.1f} km, Speed {v_mag:.0f} m/s")
            continue

    elif phase == "S1_only":
        if m_s1_prop > 0:
            s1_isp = S1_ISP_SL + (S1_ISP_VAC - S1_ISP_SL) * min(1.0, h / 50000)
            mdot_s1 = S1_THRUST / (s1_isp * 9.81)
            dm_s1 = min(mdot_s1 * dt, m_s1_prop)
            thrust = S1_THRUST
            mdot = dm_s1 / dt
            m_s1_prop -= dm_s1
            m_total -= dm_s1
        else:
            # MECO → coast → stage separation
            m_total -= S1_DRY_MASS
            phase = "coast"
            s1_separated = True
            events.append(("1段 MECO/分離", t, h/1000, v_mag, theta * R_EARTH / 1000))
            print(f"  T+{t:.0f}s: 1st stage MECO — Alt {h/1000:.1f} km, Speed {v_mag:.0f} m/s")
            coast_start = t
            continue

    elif phase == "coast":
        thrust = 0
        mdot = 0
        if t - coast_start >= COAST_BETWEEN_STAGES:
            phase = "S2"
            events.append(("2段 点火", t, h/1000, v_mag, theta * R_EARTH / 1000))
            print(f"  T+{t:.0f}s: 2nd stage ignition — Alt {h/1000:.1f} km")
            continue

    elif phase == "S2":
        # Check orbital insertion condition:
        # v_theta >= v_circular at current altitude AND h >= 250 km
        v_circ_local = math.sqrt(MU / r)
        if h > 250000 and v_theta >= v_circ_local * 0.997:
            phase = "SECO"
            events.append(("2段 SECO — 軌道投入完了", t, h/1000, v_mag, theta * R_EARTH / 1000))
            print(f"  T+{t:.0f}s: SECO — ORBIT ACHIEVED!")
            print(f"          Alt {h/1000:.1f} km, Speed {v_mag:.0f} m/s")
            print(f"          v_circ={v_circ_local:.0f} m/s, v_theta={v_theta:.0f} m/s")
            orbit_achieved = True
        elif m_s2_prop > 0:
            mdot_s2 = S2_THRUST / (S2_ISP * 9.81)
            dm_s2 = min(mdot_s2 * dt, m_s2_prop)
            thrust = S2_THRUST
            mdot = dm_s2 / dt
            m_s2_prop -= dm_s2
            m_total -= dm_s2
        else:
            phase = "SECO"
            events.append(("2段 SECO — 燃料枯渇", t, h/1000, v_mag, theta * R_EARTH / 1000))
            print(f"  T+{t:.0f}s: SECO (propellant depleted)")
            print(f"          Alt {h/1000:.1f} km, Speed {v_mag:.0f} m/s")
            orbit_achieved = v_theta >= v_circ_local * 0.90

    elif phase == "SECO":
        # Coasting in orbit — continue for a bit to show orbital arc
        thrust = 0
        mdot = 0
        if t > events[-1][1] + 300:  # 5 min of orbital coast
            break

    # Fairing jettison
    if not fairing_jettisoned and h > FAIRING_JETTISON_ALT:
        m_total -= FAIRING_MASS
        fairing_jettisoned = True
        events.append(("フェアリング分離", t, h/1000, v_mag, theta * R_EARTH / 1000))
        print(f"  T+{t:.0f}s: Fairing jettison — Alt {h/1000:.1f} km")

    # --- Pitch control (programmed pitch) ---
    pitch = programmed_pitch(t)
    pitch_rad = math.radians(pitch)

    # Thrust direction (in r, theta components)
    # pitch=0 → radial (up), pitch=90 → tangential (horizontal)
    if thrust > 0:
        thrust_r = thrust * math.cos(pitch_rad)
        thrust_theta = thrust * math.sin(pitch_rad)
    else:
        thrust_r = 0
        thrust_theta = 0

    # Drag (opposing velocity)
    if v_mag > 0:
        drag_r = -drag * v_r / v_mag
        drag_theta = -drag * v_theta / v_mag
    else:
        drag_r = 0
        drag_theta = 0

    # Equations of motion (polar coordinates)
    a_r = thrust_r / m_total + drag_r / m_total - g_local + v_theta**2 / r
    a_theta = thrust_theta / m_total + drag_theta / m_total - v_r * v_theta / r

    # Integration (Euler)
    v_r += a_r * dt
    v_theta += a_theta * dt
    r += v_r * dt
    theta += v_theta / r * dt

    # Store
    traj.append({
        "t": t,
        "h": h / 1000,       # km
        "v": v_mag,           # m/s
        "v_r": v_r,
        "v_theta": v_theta,
        "downrange": theta * R_EARTH / 1000,  # km
        "r": r,
        "theta": theta,
        "phase": phase,
        "mass": m_total,
        "drag_kN": drag / 1000,
        "pitch": pitch,
    })

    # Debug output
    if t >= next_debug:
        print(f"  T+{t:.0f}s: h={h/1000:.1f}km v={v_mag:.0f}m/s phase={phase} "
              f"m={m_total/1000:.0f}t SRBp={m_srb_prop/1000:.0f}t S1p={m_s1_prop/1000:.0f}t S2p={m_s2_prop/1000:.0f}t "
              f"pitch={pitch:.1f}° vr={v_r:.0f} vt={v_theta:.0f}")
        next_debug += debug_interval

    t += dt

# Final orbital parameters
if orbit_achieved:
    v_orbital = math.sqrt(v_r**2 + v_theta**2)
    h_final = (r - R_EARTH) / 1000
    v_circular = math.sqrt(MU / r)

    # Orbital elements
    energy = 0.5 * v_orbital**2 - MU / r
    a_orbit = -MU / (2 * energy) if energy < 0 else float('inf')
    h_angular = r * v_theta
    e_orbit = math.sqrt(1 + 2 * energy * h_angular**2 / MU**2) if energy < 0 else 1.0
    r_perigee = a_orbit * (1 - e_orbit)
    r_apogee = a_orbit * (1 + e_orbit)

    print(f"\n=== Orbital Parameters ===")
    print(f"  Altitude: {h_final:.1f} km")
    print(f"  Velocity: {v_orbital:.0f} m/s (circular: {v_circular:.0f} m/s)")
    print(f"  Semi-major axis: {a_orbit/1000:.0f} km")
    print(f"  Eccentricity: {e_orbit:.4f}")
    print(f"  Perigee: {(r_perigee - R_EARTH)/1000:.1f} km")
    print(f"  Apogee: {(r_apogee - R_EARTH)/1000:.1f} km")
    print(f"  Period: {2*math.pi*math.sqrt(a_orbit**3/MU)/60:.1f} min")

# ============================================================
# Plots
# ============================================================
times = [p["t"] for p in traj]
alts = [p["h"] for p in traj]
speeds = [p["v"] for p in traj]
downranges = [p["downrange"] for p in traj]
masses = [p["mass"]/1000 for p in traj]
drags = [p["drag_kN"] for p in traj]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Event markers
ev_times = [e[1] for e in events]
ev_alts = [e[2] for e in events]
ev_speeds = [e[3] for e in events]
ev_labels = [e[0] for e in events]
# English labels for plotting (avoid CJK font issues)
ev_labels_en = []
for lbl in ev_labels:
    if "SRB" in lbl:
        ev_labels_en.append("SRB-3 Sep")
    elif "MECO" in lbl or "1段" in lbl:
        ev_labels_en.append("MECO/S1 Sep")
    elif "2段 点火" in lbl or "2段 SECO" in lbl:
        ev_labels_en.append(lbl.replace("2段 点火", "S2 Ignition").replace("2段 SECO — 軌道投入完了", "SECO"))
    elif "フェアリング" in lbl:
        ev_labels_en.append("Fairing Sep")
    else:
        ev_labels_en.append(lbl)

# (a) Altitude vs time
ax = axes[0, 0]
ax.plot(times, alts, "b-", lw=2)
for et, ea, lbl in zip(ev_times, ev_alts, ev_labels_en):
    ax.axvline(x=et, color="gray", ls="--", alpha=0.4)
    ax.annotate(lbl, (et, ea), fontsize=7, rotation=30, ha="left")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude (km)")
ax.set_title("(a) Altitude vs Time")
ax.grid(True, alpha=0.3)

# (b) Speed vs time
ax = axes[0, 1]
ax.plot(times, speeds, "r-", lw=2)
ax.axhline(y=7800, color="green", ls="--", alpha=0.5, label="Orbital velocity (7.8 km/s)")
for et, es, lbl in zip(ev_times, ev_speeds, ev_labels):
    ax.axvline(x=et, color="gray", ls="--", alpha=0.4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("(b) Speed vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Altitude vs downrange
ax = axes[0, 2]
ax.plot(downranges, alts, "g-", lw=2)
for e, lbl_en in zip(events, ev_labels_en):
    ax.plot(e[4], e[2], "ko", ms=5)
    ax.annotate(lbl_en, (e[4], e[2]), fontsize=7, ha="left")
ax.set_xlabel("Downrange (km)")
ax.set_ylabel("Altitude (km)")
ax.set_title("(c) Trajectory Profile")
ax.grid(True, alpha=0.3)

# (d) Mass vs time
ax = axes[1, 0]
ax.plot(times, masses, "purple", lw=2)
for et, lbl in zip(ev_times, ev_labels):
    ax.axvline(x=et, color="gray", ls="--", alpha=0.4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass (ton)")
ax.set_title("(d) Vehicle Mass")
ax.grid(True, alpha=0.3)

# (e) Drag vs time
ax = axes[1, 1]
ax.plot(times, drags, "orange", lw=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Drag Force (kN)")
ax.set_title("(e) Aerodynamic Drag")
ax.grid(True, alpha=0.3)

# (f) Speed vs altitude (trajectory diagram)
ax = axes[1, 2]
ax.plot(speeds, alts, "b-", lw=2)
for e, lbl_en in zip(events, ev_labels_en):
    ax.plot(e[3], e[2], "ro", ms=5)
    ax.annotate(lbl_en, (e[3], e[2]), fontsize=7, ha="left")
ax.set_xlabel("Speed (m/s)")
ax.set_ylabel("Altitude (km)")
ax.set_title("(f) Speed-Altitude Diagram")
ax.grid(True, alpha=0.3)

plt.suptitle(
    "JAXA H3-22S — Tanegashima to LEO 300km (Full Trajectory)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("jaxa_h3_orbital_results.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: jaxa_h3_orbital_results.png")

# ============================================================
# KML Generation
# ============================================================
def latlon_from_polar(theta_val, h_val, lat0, lon0, azimuth_rad):
    """Convert downrange angle and altitude to lat/lon/alt."""
    dist_surface = theta_val * R_EARTH
    dlat = dist_surface * math.cos(azimuth_rad) / R_EARTH * (180 / math.pi)
    dlon = dist_surface * math.sin(azimuth_rad) / (R_EARTH * math.cos(math.radians(lat0))) * (180 / math.pi)
    return lat0 + dlat, lon0 + dlon, h_val

tomorrow = datetime.date.today() + datetime.timedelta(days=1)
KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)
def k(tag): return f"{{{KML_NS}}}{tag}"
def g(tag): return f"{{{GX_NS}}}{tag}"

root = ET.Element(k("kml"))
doc = ET.SubElement(root, k("Document"))
ET.SubElement(doc, k("name")).text = "JAXA H3-22S — 種子島 → LEO 軌道投入"
ET.SubElement(doc, k("open")).text = "1"

if orbit_achieved:
    desc = (
        f"JAXA H3-22S 軌道投入シミュレーション\n"
        f"種子島宇宙センター → LEO {h_final:.0f} km\n"
        f"Date: {tomorrow}\n"
        f"軌道速度: {v_orbital:.0f} m/s\n"
        f"近地点: {(r_perigee - R_EARTH)/1000:.0f} km\n"
        f"遠地点: {(r_apogee - R_EARTH)/1000:.0f} km"
    )
else:
    desc = "H3-22S trajectory simulation"
ET.SubElement(doc, k("description")).text = desc

# Styles
phase_colors = {
    "SRB+S1":   ("ff0000ff", "5"),    # Red
    "S1_only":  ("ff0066ff", "4"),    # Orange-Red
    "coast":    ("ff00ccff", "3"),    # Yellow
    "S2":       ("ffff6600", "4"),    # Blue
    "SECO":     ("ff00ff00", "3"),    # Green (orbit)
}
for pname, (col, wid) in phase_colors.items():
    st = ET.SubElement(doc, k("Style"), id=f"s_{pname}")
    ls = ET.SubElement(st, k("LineStyle"))
    ET.SubElement(ls, k("color")).text = col
    ET.SubElement(ls, k("width")).text = wid
    ps = ET.SubElement(st, k("PolyStyle"))
    ET.SubElement(ps, k("color")).text = col[:2] + "ffffff"
    ET.SubElement(ps, k("fill")).text = "1"

for iname, href, col, scl in [
    ("launch", "https://maps.google.com/mapfiles/kml/shapes/triangle.png", "ff0000ff", "1.8"),
    ("event", "https://maps.google.com/mapfiles/kml/paddle/orange-circle.png", "ff00ccff", "1.3"),
    ("orbit", "https://maps.google.com/mapfiles/kml/shapes/star.png", "ff00ff00", "1.6"),
]:
    st = ET.SubElement(doc, k("Style"), id=f"i_{iname}")
    ics = ET.SubElement(st, k("IconStyle"))
    ET.SubElement(ics, k("color")).text = col
    ET.SubElement(ics, k("scale")).text = scl
    ic = ET.SubElement(ics, k("Icon"))
    ET.SubElement(ic, k("href")).text = href
    lbs = ET.SubElement(st, k("LabelStyle"))
    ET.SubElement(lbs, k("scale")).text = "1.1"

# Camera — wide view to see full trajectory
la = ET.SubElement(doc, k("LookAt"))
ET.SubElement(la, k("longitude")).text = str(TNSC_LON + 2)
ET.SubElement(la, k("latitude")).text = str(TNSC_LAT - 1)
ET.SubElement(la, k("altitude")).text = "200000"
ET.SubElement(la, k("heading")).text = "90"
ET.SubElement(la, k("tilt")).text = "55"
ET.SubElement(la, k("range")).text = "1500000"
ET.SubElement(la, k("altitudeMode")).text = "absolute"

# Trajectory phases
tf = ET.SubElement(doc, k("Folder"))
ET.SubElement(tf, k("name")).text = "飛行フェーズ (Flight Phases)"
ET.SubElement(tf, k("open")).text = "1"

phase_names_jp = {
    "SRB+S1": "Phase 1: SRB-3 + 1段 同時燃焼",
    "S1_only": "Phase 2: 1段のみ (LE-9×2)",
    "coast": "Phase 3: コースト (段間)",
    "S2": "Phase 4: 2段燃焼 (LE-5B-3)",
    "SECO": "Phase 5: 軌道周回 (SECO後)",
}

# Group by phase
phases_grouped = {}
for pt in traj:
    ph = pt["phase"]
    if ph not in phases_grouped:
        phases_grouped[ph] = []
    phases_grouped[ph].append(pt)

# Ensure continuity
po = ["SRB+S1", "S1_only", "coast", "S2", "SECO"]
for i in range(len(po) - 1):
    if po[i] in phases_grouped and po[i+1] in phases_grouped and phases_grouped[po[i+1]]:
        phases_grouped[po[i]].append(phases_grouped[po[i+1]][0])

for pn in po:
    if pn not in phases_grouped or not phases_grouped[pn]:
        continue
    pts = phases_grouped[pn]
    pm = ET.SubElement(tf, k("Placemark"))
    ET.SubElement(pm, k("name")).text = phase_names_jp.get(pn, pn)
    ET.SubElement(pm, k("styleUrl")).text = f"#s_{pn}"

    t0, t1 = pts[0]["t"], pts[-1]["t"]
    mx_h = max(p["h"] for p in pts)
    mx_v = max(p["v"] for p in pts)
    mx_dr = max(p["downrange"] for p in pts)
    ET.SubElement(pm, k("description")).text = (
        f"Time: T+{t0:.0f} - T+{t1:.0f} s\n"
        f"Max altitude: {mx_h:.1f} km\n"
        f"Max speed: {mx_v:.0f} m/s (Mach {mx_v/343:.1f})\n"
        f"Downrange: {mx_dr:.0f} km"
    )

    ls_el = ET.SubElement(pm, k("LineString"))
    ET.SubElement(ls_el, k("extrude")).text = "1"
    ET.SubElement(ls_el, k("tessellate")).text = "1"
    ET.SubElement(ls_el, k("altitudeMode")).text = "absolute"

    # Sample every N points for KML
    step = max(1, len(pts) // 500)
    sampled = pts[::step]
    if sampled[-1] != pts[-1]:
        sampled.append(pts[-1])

    coords = []
    for p in sampled:
        lat, lon, alt = latlon_from_polar(p["theta"], p["h"] * 1000, TNSC_LAT, TNSC_LON, AZIMUTH)
        coords.append(f"{lon:.6f},{lat:.6f},{alt:.0f}")
    ET.SubElement(ls_el, k("coordinates")).text = " ".join(coords)

# Key events
ef = ET.SubElement(doc, k("Folder"))
ET.SubElement(ef, k("name")).text = "主要イベント (Key Events)"
ET.SubElement(ef, k("open")).text = "1"

# Launch site
pm = ET.SubElement(ef, k("Placemark"))
ET.SubElement(pm, k("name")).text = "種子島宇宙センター 吉信射点"
ET.SubElement(pm, k("styleUrl")).text = "#i_launch"
ET.SubElement(pm, k("description")).text = (
    f"Yoshinobu Launch Complex LP-2\n"
    f"H3-22S 打ち上げ\n"
    f"総質量: {M_LIFTOFF/1000:.0f} ton\n"
    f"推力: {(SRB_THRUST + S1_THRUST)/1e6:.1f} MN\n"
    f"T/W: {(SRB_THRUST + S1_THRUST)/(M_LIFTOFF * 9.81):.2f}"
)
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, k("coordinates")).text = f"{TNSC_LON},{TNSC_LAT},{TNSC_ELEV}"

# All events
for ev_name, ev_t, ev_h, ev_v, ev_dr in events:
    # Find theta for this event
    closest = min(traj, key=lambda p: abs(p["t"] - ev_t))
    lat, lon, alt = latlon_from_polar(closest["theta"], ev_h * 1000, TNSC_LAT, TNSC_LON, AZIMUTH)

    pm = ET.SubElement(ef, k("Placemark"))
    is_orbit = "SECO" in ev_name or "軌道" in ev_name
    ET.SubElement(pm, k("name")).text = f"T+{ev_t:.0f}s: {ev_name}"
    ET.SubElement(pm, k("styleUrl")).text = f"#i_{'orbit' if is_orbit else 'event'}"
    ET.SubElement(pm, k("description")).text = (
        f"Time: T+{ev_t:.0f} s\n"
        f"Altitude: {ev_h:.1f} km\n"
        f"Speed: {ev_v:.0f} m/s\n"
        f"Downrange: {ev_dr:.0f} km"
    )
    pt_el = ET.SubElement(pm, k("Point"))
    ET.SubElement(pt_el, k("altitudeMode")).text = "absolute"
    ET.SubElement(pt_el, k("coordinates")).text = f"{lon:.6f},{lat:.6f},{alt:.0f}"

# gx:Track animation
af = ET.SubElement(doc, k("Folder"))
ET.SubElement(af, k("name")).text = "アニメーション (Time Slider)"
ET.SubElement(af, k("open")).text = "0"
pm = ET.SubElement(af, k("Placemark"))
ET.SubElement(pm, k("name")).text = "H3 Rocket Position"
st = ET.SubElement(pm, k("Style"))
ics = ET.SubElement(st, k("IconStyle"))
ET.SubElement(ics, k("scale")).text = "1.5"
ic = ET.SubElement(ics, k("Icon"))
ET.SubElement(ic, k("href")).text = "https://maps.google.com/mapfiles/kml/shapes/rocket.png"

track = ET.SubElement(pm, g("Track"))
ET.SubElement(track, k("altitudeMode")).text = "absolute"
base_dt = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 1, 0, 0)

anim_step = max(1, len(traj) // 300)
anim_pts = traj[::anim_step]

for p in anim_pts:
    when = base_dt + datetime.timedelta(seconds=p["t"])
    ET.SubElement(track, k("when")).text = when.strftime("%Y-%m-%dT%H:%M:%SZ")
for p in anim_pts:
    lat, lon, alt = latlon_from_polar(p["theta"], p["h"] * 1000, TNSC_LAT, TNSC_LON, AZIMUTH)
    ET.SubElement(track, g("coord")).text = f"{lon:.6f} {lat:.6f} {alt:.0f}"

# Flight summary
sf = ET.SubElement(doc, k("Folder"))
ET.SubElement(sf, k("name")).text = "飛行データ (Flight Summary)"
pm = ET.SubElement(sf, k("Placemark"))
ET.SubElement(pm, k("name")).text = "H3-22S Flight Data"

orbit_info = ""
if orbit_achieved:
    orbit_info = f"""
<tr style="background:#003366;color:white;"><td colspan="2"><b>Orbital Parameters</b></td></tr>
<tr><td>Orbit altitude</td><td><b>{h_final:.0f} km</b></td></tr>
<tr><td>Orbital speed</td><td>{v_orbital:.0f} m/s</td></tr>
<tr><td>Perigee</td><td>{(r_perigee - R_EARTH)/1000:.0f} km</td></tr>
<tr><td>Apogee</td><td>{(r_apogee - R_EARTH)/1000:.0f} km</td></tr>
<tr><td>Eccentricity</td><td>{e_orbit:.4f}</td></tr>
<tr><td>Period</td><td>{2*math.pi*math.sqrt(a_orbit**3/MU)/60:.1f} min</td></tr>
"""

ET.SubElement(pm, k("description")).text = f"""<![CDATA[
<h2>JAXA H3-22S — 軌道投入シミュレーション</h2>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">
<tr style="background:#003366;color:white;"><td colspan="2"><b>Vehicle: H3-22S</b></td></tr>
<tr><td>Liftoff mass</td><td>{M_LIFTOFF/1000:.0f} ton</td></tr>
<tr><td>SRB-3 (x2)</td><td>{SRB_THRUST/1e6:.1f} MN, {SRB_BURN_TIME:.0f}s</td></tr>
<tr><td>1st stage (LE-9 x2)</td><td>{S1_THRUST/1e6:.2f} MN, Isp {S1_ISP_VAC:.0f}s</td></tr>
<tr><td>2nd stage (LE-5B-3)</td><td>{S2_THRUST/1e3:.0f} kN, Isp {S2_ISP:.0f}s</td></tr>
<tr><td>Payload</td><td>{PAYLOAD_MASS/1000:.0f} ton (LEO)</td></tr>
<tr style="background:#003366;color:white;"><td colspan="2"><b>Flight Events</b></td></tr>
{"".join(f'<tr><td>T+{e[1]:.0f}s</td><td>{e[0]} ({e[2]:.0f}km, {e[3]:.0f}m/s)</td></tr>' for e in events)}
{orbit_info}
</table>
<p><i>2D gravity-turn simulation / exponential atmosphere / Earth rotation</i></p>
]]>"""
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "clampToGround"
ET.SubElement(pt_el, k("coordinates")).text = f"{TNSC_LON + 0.01},{TNSC_LAT + 0.01},0"

# Write KML
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
output_kml = "jaxa_h3_orbital.kml"
tree.write(output_kml, xml_declaration=True, encoding="UTF-8")

print(f"\nKML saved: {output_kml}")
print("\nAll outputs:")
print(f"  {output_kml} — Google Earth で軌道投入まで可視化")
print(f"  jaxa_h3_orbital_results.png — 飛行プロファイル6面図")
