"""
JAXA H3 Rocket-Inspired Launch Simulation from Tanegashima Space Center
========================================================================
- Launch site: 種子島宇宙センター 吉信射点 (Yoshinobu Launch Complex)
- Rocket: JAXA H3-inspired high-power sounding rocket
- Motor: Cesaroni 9977 M2245 (highest total impulse available)
- Heading: East (azimuth ~90°, typical JAXA GTO trajectory)
- Generates professional KML for Google Earth
"""
import datetime
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rocketpy import Environment, Flight, Rocket, SolidMotor

# ============================================================
# 1. Environment — 種子島宇宙センター 吉信射点
# ============================================================
# Yoshinobu Launch Complex LP-2
TNSC_LAT = 30.4009  # 北緯 30°24'
TNSC_LON = 130.9751  # 東経 130°58'
TNSC_ELEV = 50       # ~50m ASL (sea cliff)

env = Environment(latitude=TNSC_LAT, longitude=TNSC_LON, elevation=TNSC_ELEV)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 1))  # 10:00 JST = 01:00 UTC
env.set_atmospheric_model(type="standard_atmosphere")

print("=" * 60)
print("  JAXA H3-Inspired Launch Simulation")
print("  種子島宇宙センター 吉信射点")
print("=" * 60)
print(f"  Date: {tomorrow} 10:00 JST")
print(f"  Location: {TNSC_LAT}°N, {TNSC_LON}°E, {TNSC_ELEV}m ASL")

# ============================================================
# 2. Motor — 大型固体ロケットモーター
#    Cesaroni 9977 M2245 (highest impulse: 9977 Ns, ~3000N peak)
# ============================================================
motor = SolidMotor(
    thrust_source="data/motors/cesaroni/Cesaroni_9977M2245-P.eng",
    dry_mass=2.8,
    dry_inertia=(0.2, 0.2, 0.005),
    nozzle_radius=37.5 / 1000,   # 75mm motor casing
    grain_number=6,
    grain_density=1820,
    grain_outer_radius=37.5 / 1000,
    grain_initial_inner_radius=17 / 1000,
    grain_initial_height=150 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.5,
    center_of_dry_mass_position=0.4,
    nozzle_position=0,
    burn_time=4.4,
    throat_radius=13 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

print(f"\n=== Motor: Cesaroni M2245 ===")
print(f"  Total impulse: {motor.total_impulse:.0f} Ns")
print(f"  Burn time: {motor.burn_out_time:.2f} s")
print(f"  Average thrust: {motor.average_thrust:.0f} N")

# ============================================================
# 3. Rocket — H3ロケットをイメージした大型機体
#    (RocketPy の物理範囲内で最大限スケールアップ)
# ============================================================
rocket = Rocket(
    radius=160 / 2000,        # 160mm diameter (large HPR)
    mass=22.0,                 # 22 kg dry mass
    inertia=(12.0, 12.0, 0.06),
    power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
    power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

rocket.set_rail_buttons(
    upper_button_position=0.1,
    lower_button_position=-0.8,
    angular_position=45,
)

rocket.add_motor(motor, position=-1.6)

# Nose cone — Von Karman (H3 フェアリング風)
nose = rocket.add_nose(length=0.75, kind="vonKarman", position=1.6)

# Fins — 4枚 (安定翼)
fins = rocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.200,
    tip_chord=0.080,
    span=0.150,
    position=-1.35,
    cant_angle=0,
    airfoil=("data/airfoils/NACA0012-radians.txt", "radians"),
)

# Tail
tail = rocket.add_tail(
    top_radius=0.080, bottom_radius=0.055, length=0.080, position=-1.50
)

# Parachutes — Drogue at apogee, Main at 500m AGL
rocket.add_parachute(
    "Drogue", cd_s=1.5, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)
rocket.add_parachute(
    "Main", cd_s=12.0, trigger=500, sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)

print(f"\n=== Rocket: H3-Inspired Sounding Rocket ===")
print(f"  Dry mass: {rocket.mass:.1f} kg")
print(f"  Diameter: {rocket.radius*2*1000:.0f} mm")

# ============================================================
# 4. Flight Simulation
#    方位角: 110° (南東方向、太平洋側へ — JAXA の典型的な打ち上げ方向)
#    射角: 84° (ほぼ垂直)
# ============================================================
LAUNCH_HEADING = 110    # SE toward Pacific (JAXA typical)
LAUNCH_INCLINATION = 84  # 6° from vertical

print(f"\n=== Flight Simulation ===")
print(f"  Heading: {LAUNCH_HEADING}° (SE, toward Pacific)")
print(f"  Inclination: {LAUNCH_INCLINATION}° from horizontal")

flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=8.0,       # Longer rail for heavier rocket
    inclination=LAUNCH_INCLINATION,
    heading=LAUNCH_HEADING,
)

apogee_agl = flight.apogee - TNSC_ELEV
print(f"\n  --- Results ---")
print(f"  Apogee (AGL): {apogee_agl:.0f} m")
print(f"  Apogee (ASL): {flight.apogee:.0f} m")
print(f"  Apogee time: {flight.apogee_time:.1f} s")
print(f"  Max speed: {flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})")
print(f"  Max acceleration: {flight.max_acceleration:.0f} m/s² ({flight.max_acceleration/9.81:.1f} G)")
print(f"  Out-of-rail speed: {flight.out_of_rail_velocity:.1f} m/s")
print(f"  Impact velocity: {flight.impact_velocity:.1f} m/s")
print(f"  Flight time: {flight.t_final:.0f} s")
print(f"  Landing: ({flight.x_impact:.0f}, {flight.y_impact:.0f}) m from launch")

# ============================================================
# 5. Plot Results
# ============================================================
t = np.linspace(0, flight.t_final, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Altitude
ax = axes[0, 0]
alt = [flight.altitude(ti) - TNSC_ELEV for ti in t]
ax.plot(t, alt, "b-", linewidth=1.5)
ax.axhline(y=apogee_agl, color="r", ls="--", alpha=0.5, label=f"Apogee: {apogee_agl:.0f} m")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude AGL (m)")
ax.set_title("(a) Altitude Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Speed
ax = axes[0, 1]
speed = [flight.speed(ti) for ti in t]
ax.plot(t, speed, "r-", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("(b) Speed Profile")
ax.grid(True, alpha=0.3)

# (c) Trajectory X-Z
ax = axes[1, 0]
x_pos = [flight.x(ti) for ti in t]
z_pos = [flight.altitude(ti) - TNSC_ELEV for ti in t]
ax.plot(x_pos, z_pos, "g-", linewidth=1.5)
ax.set_xlabel("Downrange East (m)")
ax.set_ylabel("Altitude AGL (m)")
ax.set_title("(c) Trajectory (Side View)")
ax.grid(True, alpha=0.3)

# (d) Ground Track
ax = axes[1, 1]
y_pos = [flight.y(ti) for ti in t]
ax.plot(x_pos, y_pos, "m-", linewidth=1.5)
ax.plot(0, 0, "r^", ms=10, label="Yoshinobu LP-2")
ax.plot(flight.x_impact, flight.y_impact, "kx", ms=10, label="Landing")
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_title("(d) Ground Track")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.suptitle(
    "JAXA H3-Inspired Launch — Tanegashima Space Center (6-DOF)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("jaxa_tanegashima_results.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: jaxa_tanegashima_results.png")

# 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_pos, y_pos, z_pos, "b-", linewidth=1.5)
ax.plot([0], [0], [0], "r^", ms=10, label="Yoshinobu LP-2")
ax.plot([flight.x_impact], [flight.y_impact], [0], "kx", ms=10, label="Landing")
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_zlabel("Altitude AGL (m)")
ax.set_title("3D Trajectory — JAXA Tanegashima Launch")
ax.legend()
plt.savefig("jaxa_tanegashima_3d.png", dpi=150, bbox_inches="tight")
print("Figure saved: jaxa_tanegashima_3d.png")

# ============================================================
# 6. Generate Professional KML
# ============================================================
R_earth = 6371000.0

def meters_to_latlon(x, y, lat0, lon0):
    dlat = y / R_earth * (180.0 / math.pi)
    dlon = x / (R_earth * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon

# Sample trajectory
dt_sample = 0.5
t_vals = np.arange(0, flight.t_final + dt_sample, dt_sample)
if t_vals[-1] > flight.t_final:
    t_vals[-1] = flight.t_final

trajectory = []
for tv in t_vals:
    x = float(flight.x(tv))
    y = float(flight.y(tv))
    alt_val = float(flight.altitude(tv))
    spd = float(flight.speed(tv))
    lat, lon = meters_to_latlon(x, y, TNSC_LAT, TNSC_LON)
    trajectory.append({"t": float(tv), "lat": lat, "lon": lon, "alt": alt_val, "speed": spd})

burn_time = float(motor.burn_out_time)
apogee_time = float(flight.apogee_time)
apogee_alt = float(flight.apogee)
drogue_time = apogee_time

# Main deploy altitude
main_deploy_alt = TNSC_ELEV + 500.0
main_time = None
for i in range(len(trajectory) - 1):
    if (trajectory[i]["t"] > apogee_time
        and trajectory[i]["alt"] > main_deploy_alt
        and trajectory[i+1]["alt"] <= main_deploy_alt):
        main_time = trajectory[i+1]["t"]
        break
if main_time is None:
    main_time = flight.t_final * 0.8

# --- Build KML ---
KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)

def kml(tag): return f"{{{KML_NS}}}{tag}"
def gx(tag): return f"{{{GX_NS}}}{tag}"

root = ET.Element(kml("kml"))
doc = ET.SubElement(root, kml("Document"))
ET.SubElement(doc, kml("name")).text = "JAXA H3-Inspired Launch — 種子島宇宙センター"
ET.SubElement(doc, kml("open")).text = "1"

desc_text = (
    f"JAXA H3ロケット風 打ち上げシミュレーション\n"
    f"Launch Site: 種子島宇宙センター 吉信射点 (Yoshinobu LP-2)\n"
    f"Date: {tomorrow} 10:00 JST\n"
    f"Heading: {LAUNCH_HEADING}° (SE, Pacific)\n"
    f"Apogee: {apogee_agl:.0f} m AGL / {apogee_alt:.0f} m ASL\n"
    f"Max Speed: {flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})\n"
    f"Flight Time: {flight.t_final:.0f} s"
)
ET.SubElement(doc, kml("description")).text = desc_text

# --- Styles ---
styles = {
    "powered":  ("ff0000ff", "5"),    # Red
    "coast":    ("ff00a5ff", "3.5"),   # Orange
    "drogue":   ("ffffcc00", "3"),     # Cyan-blue
    "main":     ("ff00ff00", "2.5"),   # Green
}
for name, (color, width) in styles.items():
    st = ET.SubElement(doc, kml("Style"), id=f"s_{name}")
    ls = ET.SubElement(st, kml("LineStyle"))
    ET.SubElement(ls, kml("color")).text = color
    ET.SubElement(ls, kml("width")).text = width
    ps = ET.SubElement(st, kml("PolyStyle"))
    ET.SubElement(ps, kml("color")).text = color[:2] + "ffffff"
    ET.SubElement(ps, kml("fill")).text = "1"

pm_icons = {
    "launch": ("https://maps.google.com/mapfiles/kml/shapes/triangle.png", "ff0000ff", "1.6"),
    "apogee": ("https://maps.google.com/mapfiles/kml/shapes/star.png", "ff00ffff", "1.5"),
    "deploy": ("https://maps.google.com/mapfiles/kml/paddle/ylw-circle.png", "ff00ffff", "1.2"),
    "landing": ("https://maps.google.com/mapfiles/kml/shapes/cross-hairs.png", "ff00ff00", "1.4"),
}
for name, (href, color, scale) in pm_icons.items():
    st = ET.SubElement(doc, kml("Style"), id=f"icon_{name}")
    ics = ET.SubElement(st, kml("IconStyle"))
    ET.SubElement(ics, kml("color")).text = color
    ET.SubElement(ics, kml("scale")).text = scale
    ic = ET.SubElement(ics, kml("Icon"))
    ET.SubElement(ic, kml("href")).text = href
    lbs = ET.SubElement(st, kml("LabelStyle"))
    ET.SubElement(lbs, kml("scale")).text = "1.2"
    ET.SubElement(lbs, kml("color")).text = "ffffffff"

# --- Camera ---
la = ET.SubElement(doc, kml("LookAt"))
ET.SubElement(la, kml("longitude")).text = str(TNSC_LON)
ET.SubElement(la, kml("latitude")).text = str(TNSC_LAT)
ET.SubElement(la, kml("altitude")).text = str(int(apogee_alt / 2))
ET.SubElement(la, kml("heading")).text = "110"
ET.SubElement(la, kml("tilt")).text = "65"
ET.SubElement(la, kml("range")).text = str(int(apogee_alt * 2))
ET.SubElement(la, kml("altitudeMode")).text = "absolute"

# --- Trajectory Phases ---
tf = ET.SubElement(doc, kml("Folder"))
ET.SubElement(tf, kml("name")).text = "飛行フェーズ (Trajectory Phases)"
ET.SubElement(tf, kml("open")).text = "1"

def get_phase(t_val):
    if t_val <= burn_time:
        return "powered"
    elif t_val <= drogue_time:
        return "coast"
    elif t_val <= main_time:
        return "drogue"
    return "main"

phase_labels = {
    "powered": "Phase 1: 推進飛行 (Powered Flight)",
    "coast":   "Phase 2: 慣性飛行 (Ballistic Coast)",
    "drogue":  "Phase 3: ドローグ降下 (Drogue Descent)",
    "main":    "Phase 4: メインパラシュート降下 (Main Chute)",
}

phases_data = {"powered": [], "coast": [], "drogue": [], "main": []}
for pt in trajectory:
    phases_data[get_phase(pt["t"])].append(pt)

phase_order = ["powered", "coast", "drogue", "main"]
for i in range(len(phase_order) - 1):
    c, n = phase_order[i], phase_order[i+1]
    if phases_data[n]:
        phases_data[c].append(phases_data[n][0])

for pn in phase_order:
    pts = phases_data[pn]
    if not pts:
        continue
    pm = ET.SubElement(tf, kml("Placemark"))
    ET.SubElement(pm, kml("name")).text = phase_labels[pn]
    ET.SubElement(pm, kml("styleUrl")).text = f"#s_{pn}"
    t0, t1 = pts[0]["t"], pts[-1]["t"]
    mx_alt = max(p["alt"] - TNSC_ELEV for p in pts)
    mx_spd = max(p["speed"] for p in pts)
    ET.SubElement(pm, kml("description")).text = (
        f"Time: {t0:.1f} - {t1:.1f} s\n"
        f"Max altitude: {mx_alt:.0f} m AGL\n"
        f"Max speed: {mx_spd:.0f} m/s"
    )
    ls_elem = ET.SubElement(pm, kml("LineString"))
    ET.SubElement(ls_elem, kml("extrude")).text = "1"
    ET.SubElement(ls_elem, kml("tessellate")).text = "1"
    ET.SubElement(ls_elem, kml("altitudeMode")).text = "absolute"
    ET.SubElement(ls_elem, kml("coordinates")).text = " ".join(
        f"{p['lon']:.8f},{p['lat']:.8f},{p['alt']:.1f}" for p in pts
    )

# --- Key Events ---
ef = ET.SubElement(doc, kml("Folder"))
ET.SubElement(ef, kml("name")).text = "主要イベント (Key Events)"
ET.SubElement(ef, kml("open")).text = "1"

# Launch
pm = ET.SubElement(ef, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = "種子島宇宙センター 吉信射点"
ET.SubElement(pm, kml("styleUrl")).text = "#icon_launch"
ET.SubElement(pm, kml("description")).text = (
    f"Yoshinobu Launch Complex LP-2\n"
    f"Tanegashima Space Center, JAXA\n"
    f"Lat: {TNSC_LAT}°N, Lon: {TNSC_LON}°E\n"
    f"Elevation: {TNSC_ELEV} m ASL\n"
    f"Heading: {LAUNCH_HEADING}° (SE)\n"
    f"Inclination: {LAUNCH_INCLINATION}°\n"
    f"Rail exit speed: {flight.out_of_rail_velocity:.1f} m/s"
)
pt_el = ET.SubElement(pm, kml("Point"))
ET.SubElement(pt_el, kml("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, kml("coordinates")).text = f"{TNSC_LON},{TNSC_LAT},{TNSC_ELEV}"

# Apogee
ap_pt = min(trajectory, key=lambda p: abs(p["t"] - apogee_time))
pm = ET.SubElement(ef, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = f"最高到達点 (Apogee): {apogee_agl:.0f} m"
ET.SubElement(pm, kml("styleUrl")).text = "#icon_apogee"
ET.SubElement(pm, kml("description")).text = (
    f"Altitude: {apogee_alt:.0f} m ASL ({apogee_agl:.0f} m AGL)\n"
    f"Time: T+{apogee_time:.1f} s\n"
    f"Drogue parachute deployed"
)
pt_el = ET.SubElement(pm, kml("Point"))
ET.SubElement(pt_el, kml("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, kml("coordinates")).text = f"{ap_pt['lon']:.8f},{ap_pt['lat']:.8f},{apogee_alt:.1f}"

# Main deploy
mn_pt = min(trajectory, key=lambda p: abs(p["t"] - main_time))
pm = ET.SubElement(ef, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = "メインパラシュート展開 (500m AGL)"
ET.SubElement(pm, kml("styleUrl")).text = "#icon_deploy"
ET.SubElement(pm, kml("description")).text = (
    f"Altitude: ~{main_deploy_alt:.0f} m ASL (500 m AGL)\n"
    f"Time: T+{main_time:.1f} s\n"
    f"Main parachute (Cd*S=12.0)"
)
pt_el = ET.SubElement(pm, kml("Point"))
ET.SubElement(pt_el, kml("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, kml("coordinates")).text = f"{mn_pt['lon']:.8f},{mn_pt['lat']:.8f},{mn_pt['alt']:.1f}"

# Landing
ld_pt = trajectory[-1]
dist = math.sqrt(flight.x_impact**2 + flight.y_impact**2)
pm = ET.SubElement(ef, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = f"着水点 (Splashdown) — {dist:.0f}m"
ET.SubElement(pm, kml("styleUrl")).text = "#icon_landing"
ET.SubElement(pm, kml("description")).text = (
    f"Impact velocity: {flight.impact_velocity:.1f} m/s\n"
    f"Distance from pad: {dist:.0f} m\n"
    f"Offset: E {flight.x_impact:.0f} m, N {flight.y_impact:.0f} m\n"
    f"Time: T+{flight.t_final:.0f} s\n"
    f"太平洋上に着水 (Splashdown in Pacific)"
)
pt_el = ET.SubElement(pm, kml("Point"))
ET.SubElement(pt_el, kml("altitudeMode")).text = "clampToGround"
ET.SubElement(pt_el, kml("coordinates")).text = f"{ld_pt['lon']:.8f},{ld_pt['lat']:.8f},{TNSC_ELEV}"

# --- gx:Track Animation ---
af = ET.SubElement(doc, kml("Folder"))
ET.SubElement(af, kml("name")).text = "アニメーション (Time Animation)"
ET.SubElement(af, kml("open")).text = "0"

pm = ET.SubElement(af, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = "Rocket Position"
st = ET.SubElement(pm, kml("Style"))
ics = ET.SubElement(st, kml("IconStyle"))
ET.SubElement(ics, kml("scale")).text = "1.3"
ic = ET.SubElement(ics, kml("Icon"))
ET.SubElement(ic, kml("href")).text = "https://maps.google.com/mapfiles/kml/shapes/rocket.png"

track = ET.SubElement(pm, gx("Track"))
ET.SubElement(track, kml("altitudeMode")).text = "absolute"

base_dt = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 1, 0, 0)
anim_t = np.arange(0, flight.t_final + 1.0, 1.0)
if anim_t[-1] > flight.t_final:
    anim_t[-1] = flight.t_final

for tv in anim_t:
    when = base_dt + datetime.timedelta(seconds=float(tv))
    ET.SubElement(track, kml("when")).text = when.strftime("%Y-%m-%dT%H:%M:%SZ")
for tv in anim_t:
    x = float(flight.x(tv))
    y = float(flight.y(tv))
    alt_v = float(flight.altitude(tv))
    lat, lon = meters_to_latlon(x, y, TNSC_LAT, TNSC_LON)
    ET.SubElement(track, gx("coord")).text = f"{lon:.8f} {lat:.8f} {alt_v:.1f}"

# --- Flight Summary (HTML popup) ---
sf = ET.SubElement(doc, kml("Folder"))
ET.SubElement(sf, kml("name")).text = "飛行データ (Flight Summary)"
ET.SubElement(sf, kml("open")).text = "0"
pm = ET.SubElement(sf, kml("Placemark"))
ET.SubElement(pm, kml("name")).text = "Flight Data Sheet"
ET.SubElement(pm, kml("description")).text = f"""<![CDATA[
<h2>JAXA H3-Inspired Sounding Rocket</h2>
<h3>種子島宇宙センター 吉信射点からの打ち上げ</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">
<tr style="background:#003366;color:white;"><td colspan="2"><b>Launch Site</b></td></tr>
<tr><td>Site</td><td>Tanegashima Space Center, Yoshinobu LP-2</td></tr>
<tr><td>Coordinates</td><td>{TNSC_LAT}°N, {TNSC_LON}°E</td></tr>
<tr><td>Date</td><td>{tomorrow} 10:00 JST</td></tr>
<tr><td>Heading</td><td>{LAUNCH_HEADING}° (SE, Pacific)</td></tr>
<tr style="background:#003366;color:white;"><td colspan="2"><b>Vehicle</b></td></tr>
<tr><td>Motor</td><td>Cesaroni M2245 (9977 Ns)</td></tr>
<tr><td>Burn Time</td><td>{burn_time:.2f} s</td></tr>
<tr><td>Dry Mass</td><td>{rocket.mass:.1f} kg</td></tr>
<tr style="background:#003366;color:white;"><td colspan="2"><b>Flight Results</b></td></tr>
<tr><td>Apogee (AGL)</td><td><b>{apogee_agl:.0f} m</b></td></tr>
<tr><td>Apogee (ASL)</td><td>{apogee_alt:.0f} m</td></tr>
<tr><td>Max Speed</td><td>{flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})</td></tr>
<tr><td>Max Acceleration</td><td>{flight.max_acceleration:.0f} m/s² ({flight.max_acceleration/9.81:.1f} G)</td></tr>
<tr><td>Rail Exit Speed</td><td>{flight.out_of_rail_velocity:.1f} m/s</td></tr>
<tr><td>Impact Velocity</td><td>{abs(flight.impact_velocity):.1f} m/s</td></tr>
<tr><td>Flight Time</td><td>{flight.t_final:.0f} s</td></tr>
<tr><td>Landing Distance</td><td>{dist:.0f} m (Pacific Ocean)</td></tr>
</table>
<p><i>6-DOF Simulation by RocketPy / Standard Atmosphere</i></p>
]]>"""
pt_el = ET.SubElement(pm, kml("Point"))
ET.SubElement(pt_el, kml("altitudeMode")).text = "clampToGround"
ET.SubElement(pt_el, kml("coordinates")).text = f"{TNSC_LON + 0.003},{TNSC_LAT + 0.003},{TNSC_ELEV}"

# Write KML
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
output_kml = "jaxa_tanegashima.kml"
tree.write(output_kml, xml_declaration=True, encoding="UTF-8")

print(f"\nKML saved: {output_kml}")
print("Google Earth で開いてください！")
print(f"\nAll outputs:")
print(f"  {output_kml}")
print(f"  jaxa_tanegashima_results.png")
print(f"  jaxa_tanegashima_3d.png")
