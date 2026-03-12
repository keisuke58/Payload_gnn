"""
JAXA S-520 Sounding Rocket Simulation from Uchinoura/Tanegashima
================================================================
S-520 Specifications (JAXA ISAS):
  - Height: 9.0 m, Diameter: 0.52 m
  - Gross mass: 2,300 kg
  - Thrust: 143 kN, Burn time: 29 s
  - Apogee: ~300-430 km
  - Single-stage solid motor

RocketPy supports custom thrust curves, so we model S-520 directly.
Launch site: 内之浦宇宙空間観測所 (Uchinoura Space Center)
  - JAXA's sounding rocket launch site
  - 31.251°N, 131.082°E
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
# 1. Environment — 内之浦宇宙空間観測所
# ============================================================
# Uchinoura Space Center (USC), Kagoshima
USC_LAT = 31.2510
USC_LON = 131.0821
USC_ELEV = 230  # ~230m ASL (hilltop launch pad)

env = Environment(latitude=USC_LAT, longitude=USC_LON, elevation=USC_ELEV)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 1))
env.set_atmospheric_model(type="standard_atmosphere")

print("=" * 60)
print("  JAXA S-520 観測ロケット シミュレーション")
print("  内之浦宇宙空間観測所 (Uchinoura Space Center)")
print("=" * 60)
print(f"  Date: {tomorrow} 10:00 JST")
print(f"  Location: {USC_LAT}°N, {USC_LON}°E, {USC_ELEV}m ASL")

# ============================================================
# 2. Motor — S-520 固体ロケットモーター
#    カスタム推力曲線で S-520 を再現
#    推力: 143 kN, 燃焼時間: 29 s
#    推進薬質量: ~1,430 kg (total 2,300kg - structure ~870kg)
# ============================================================
# S-520 thrust profile (simplified from public data)
# Progressive burn: ramps up, sustains, then tails off
s520_thrust_time = [
    0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0,
    24.0, 26.0, 27.5, 28.5, 29.0, 29.5
]
s520_thrust_force = [
    0, 120000, 140000, 155000, 160000, 158000, 152000, 148000, 143000,
    135000, 125000, 100000, 60000, 25000, 0
]
# Total impulse ≈ 143kN × 29s ≈ 4,147,000 Ns (realistic for S-520)

# Write custom thrust curve to CSV
thrust_csv = "/home/nishioka/RocketPy/data/motors/s520_thrust.csv"
with open(thrust_csv, "w") as f:
    f.write("time(s),thrust(N)\n")
    for t, thr in zip(s520_thrust_time, s520_thrust_force):
        f.write(f"{t},{thr}\n")

motor = SolidMotor(
    thrust_source=thrust_csv,
    dry_mass=350,           # Motor casing ~350 kg
    dry_inertia=(800, 800, 20),
    nozzle_radius=0.15,     # ~300mm nozzle exit diameter
    grain_number=1,          # Single grain (simplified)
    grain_density=1780,      # HTPB composite propellant
    grain_outer_radius=0.24, # 480mm outer diameter
    grain_initial_inner_radius=0.08,
    grain_initial_height=5.0, # 5m grain length
    grain_separation=0,
    grains_center_of_mass_position=3.0,
    center_of_dry_mass_position=2.5,
    nozzle_position=0,
    burn_time=29.5,
    throat_radius=0.07,      # ~140mm throat
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

print(f"\n=== Motor: S-520 Solid Motor ===")
print(f"  Total impulse: {motor.total_impulse:.0f} Ns ({motor.total_impulse/1e6:.2f} MNs)")
print(f"  Burn time: {motor.burn_out_time:.1f} s")
print(f"  Average thrust: {motor.average_thrust:.0f} N ({motor.average_thrust/1000:.1f} kN)")

# ============================================================
# 3. Rocket — S-520 本体
#    全長 9.0m, 直径 0.52m, 総質量 2,300kg
# ============================================================
# Propellant mass = total_impulse / (Isp * g0)
# S-520 Isp ≈ 260s → propellant ≈ 4.15e6 / (260*9.81) ≈ 1,627 kg
# Dry mass = 2300 - 1627 ≈ 673 kg (structure + payload ~100kg)

rocket = Rocket(
    radius=0.52 / 2,        # 520mm diameter
    mass=570,                # Structural mass without motor (payload + structure)
    inertia=(2500, 2500, 30),
    power_off_drag=0.4,      # Constant Cd (reasonable for sounding rocket)
    power_on_drag=0.35,      # Slightly lower during burn
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

rocket.set_rail_buttons(
    upper_button_position=1.0,
    lower_button_position=-2.0,
    angular_position=45,
)

rocket.add_motor(motor, position=-3.5)

# Ogive nose cone
nose = rocket.add_nose(length=1.5, kind="ogive", position=4.5)

# 4 stabilization fins
fins = rocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.8,
    tip_chord=0.3,
    span=0.4,
    position=-3.2,
    cant_angle=1.5,  # Spin stabilization (S-520 is spin-stabilized)
)

# No parachute — S-520 is expendable (splashes down in Pacific)
# (impact in ocean, no recovery)

print(f"\n=== Rocket: JAXA S-520 ===")
print(f"  Total mass at launch: ~{rocket.mass + motor.propellant_initial_mass + motor.dry_mass:.0f} kg")
print(f"  Diameter: {rocket.radius*2*1000:.0f} mm")
print(f"  Length: 9.0 m")

# ============================================================
# 4. Flight Simulation
#    内之浦の打ち上げ: 方位角 131° (SE, 太平洋方向)
#    射角: 80° (観測ロケットは高高度到達のためほぼ垂直)
# ============================================================
LAUNCH_HEADING = 131     # SE toward Pacific
LAUNCH_INCLINATION = 80  # Near-vertical for altitude

print(f"\n=== Flight Simulation ===")
print(f"  Heading: {LAUNCH_HEADING}° (SE, Pacific)")
print(f"  Inclination: {LAUNCH_INCLINATION}° from horizontal")
print("  Computing 6-DOF trajectory... (this may take a moment)")

flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=15.0,        # S-520 uses a rail launcher
    inclination=LAUNCH_INCLINATION,
    heading=LAUNCH_HEADING,
    max_time=600,            # Up to 10 minutes of flight
    terminate_on_apogee=False,
)

apogee_agl = flight.apogee - USC_ELEV
print(f"\n  --- Results ---")
print(f"  Apogee (AGL): {apogee_agl/1000:.1f} km")
print(f"  Apogee (ASL): {flight.apogee/1000:.1f} km")
print(f"  Apogee time: {flight.apogee_time:.1f} s")
print(f"  Max speed: {flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})")
print(f"  Max acceleration: {flight.max_acceleration:.0f} m/s² ({flight.max_acceleration/9.81:.1f} G)")
print(f"  Out-of-rail speed: {flight.out_of_rail_velocity:.1f} m/s")
print(f"  Flight time: {flight.t_final:.0f} s")

if hasattr(flight, 'x_impact') and flight.x_impact is not None:
    dist = math.sqrt(flight.x_impact**2 + flight.y_impact**2)
    print(f"  Impact velocity: {flight.impact_velocity:.0f} m/s")
    print(f"  Impact distance: {dist/1000:.1f} km from launch")
    print(f"  Impact offset: E {flight.x_impact/1000:.1f} km, N {flight.y_impact/1000:.1f} km")
else:
    dist = 0
    print("  (Flight terminated before impact)")

# ============================================================
# 5. Plots
# ============================================================
t = np.linspace(0, flight.t_final, 2000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Altitude
ax = axes[0, 0]
alt = [flight.altitude(ti)/1000 for ti in t]
ax.plot(t, alt, "b-", lw=1.5)
ax.axhline(y=apogee_agl/1000, color="r", ls="--", alpha=0.5,
           label=f"Apogee: {apogee_agl/1000:.1f} km")
ax.axvline(x=motor.burn_out_time, color="orange", ls=":", alpha=0.7, label="Burnout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude (km)")
ax.set_title("(a) Altitude Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Speed
ax = axes[0, 1]
speed = [flight.speed(ti) for ti in t]
ax.plot(t, speed, "r-", lw=1.5)
ax.axvline(x=motor.burn_out_time, color="orange", ls=":", alpha=0.7, label="Burnout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("(b) Speed Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# (c) Trajectory side view
ax = axes[1, 0]
x_pos = [flight.x(ti)/1000 for ti in t]
z_pos = [(flight.altitude(ti) - USC_ELEV)/1000 for ti in t]
ax.plot(x_pos, z_pos, "g-", lw=1.5)
ax.set_xlabel("Downrange (km)")
ax.set_ylabel("Altitude (km)")
ax.set_title("(c) Trajectory (Side View)")
ax.grid(True, alpha=0.3)

# (d) Acceleration
ax = axes[1, 1]
# Use Mach number instead for more interesting plot
mach = [flight.mach_number(ti) for ti in t]
ax.plot(t, mach, "purple", lw=1.5)
ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5, label="Mach 1")
ax.axvline(x=motor.burn_out_time, color="orange", ls=":", alpha=0.7, label="Burnout")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mach Number")
ax.set_title("(d) Mach Number")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(
    "JAXA S-520 Sounding Rocket — Uchinoura Space Center (6-DOF)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("jaxa_s520_results.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: jaxa_s520_results.png")

# 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax3 = fig.add_subplot(111, projection="3d")
x3 = [flight.x(ti)/1000 for ti in t]
y3 = [flight.y(ti)/1000 for ti in t]
z3 = [(flight.altitude(ti) - USC_ELEV)/1000 for ti in t]
ax3.plot(x3, y3, z3, "b-", lw=1.5)
ax3.plot([0], [0], [0], "r^", ms=10, label="Uchinoura")
if flight.x_impact is not None:
    ax3.plot([flight.x_impact/1000], [flight.y_impact/1000], [0], "kx", ms=10, label="Impact")
ax3.set_xlabel("East (km)")
ax3.set_ylabel("North (km)")
ax3.set_zlabel("Altitude (km)")
ax3.set_title("3D Trajectory — JAXA S-520")
ax3.legend()
plt.savefig("jaxa_s520_3d.png", dpi=150, bbox_inches="tight")
print("Figure saved: jaxa_s520_3d.png")

# ============================================================
# 6. KML for Google Earth
# ============================================================
R_earth = 6371000.0

def meters_to_latlon(x, y, lat0, lon0):
    dlat = y / R_earth * (180.0 / math.pi)
    dlon = x / (R_earth * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon

dt_sample = 1.0
t_vals = np.arange(0, flight.t_final + dt_sample, dt_sample)
if t_vals[-1] > flight.t_final:
    t_vals[-1] = flight.t_final

trajectory = []
for tv in t_vals:
    x = float(flight.x(tv))
    y = float(flight.y(tv))
    alt_v = float(flight.altitude(tv))
    spd = float(flight.speed(tv))
    lat, lon = meters_to_latlon(x, y, USC_LAT, USC_LON)
    trajectory.append({"t": float(tv), "lat": lat, "lon": lon, "alt": alt_v, "speed": spd})

burn_time = float(motor.burn_out_time)
apogee_time = float(flight.apogee_time)
apogee_alt_val = float(flight.apogee)

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)
def k(tag): return f"{{{KML_NS}}}{tag}"
def g(tag): return f"{{{GX_NS}}}{tag}"

root = ET.Element(k("kml"))
doc = ET.SubElement(root, k("Document"))
ET.SubElement(doc, k("name")).text = "JAXA S-520 観測ロケット — 内之浦宇宙空間観測所"
ET.SubElement(doc, k("open")).text = "1"
ET.SubElement(doc, k("description")).text = (
    f"JAXA S-520 Sounding Rocket Simulation\n"
    f"内之浦宇宙空間観測所 (Uchinoura Space Center)\n"
    f"Date: {tomorrow} 10:00 JST\n"
    f"Apogee: {apogee_agl/1000:.1f} km\n"
    f"Max Speed: {flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.1f})\n"
    f"Flight Time: {flight.t_final:.0f} s"
)

# Styles
for sname, color, width in [
    ("powered", "ff0000ff", "5"), ("coast_up", "ff00a5ff", "4"),
    ("coast_down", "ffffcc00", "3.5"), ("terminal", "ff00ff00", "3"),
]:
    st = ET.SubElement(doc, k("Style"), id=f"s_{sname}")
    ls = ET.SubElement(st, k("LineStyle"))
    ET.SubElement(ls, k("color")).text = color
    ET.SubElement(ls, k("width")).text = width
    ps = ET.SubElement(st, k("PolyStyle"))
    ET.SubElement(ps, k("color")).text = color[:2] + "ffffff"
    ET.SubElement(ps, k("fill")).text = "1"

for iname, href, color, scale in [
    ("launch", "https://maps.google.com/mapfiles/kml/shapes/triangle.png", "ff0000ff", "1.6"),
    ("apogee", "https://maps.google.com/mapfiles/kml/shapes/star.png", "ff00ffff", "1.5"),
    ("impact", "https://maps.google.com/mapfiles/kml/shapes/cross-hairs.png", "ff0000ff", "1.4"),
    ("burnout", "https://maps.google.com/mapfiles/kml/paddle/orange-circle.png", "ff00a5ff", "1.2"),
]:
    st = ET.SubElement(doc, k("Style"), id=f"i_{iname}")
    ics = ET.SubElement(st, k("IconStyle"))
    ET.SubElement(ics, k("color")).text = color
    ET.SubElement(ics, k("scale")).text = scale
    ic = ET.SubElement(ics, k("Icon"))
    ET.SubElement(ic, k("href")).text = href
    lbs = ET.SubElement(st, k("LabelStyle"))
    ET.SubElement(lbs, k("scale")).text = "1.2"

# Camera
la = ET.SubElement(doc, k("LookAt"))
ET.SubElement(la, k("longitude")).text = str(USC_LON + 0.1)
ET.SubElement(la, k("latitude")).text = str(USC_LAT - 0.05)
ET.SubElement(la, k("altitude")).text = str(int(apogee_alt_val / 3))
ET.SubElement(la, k("heading")).text = "131"
ET.SubElement(la, k("tilt")).text = "70"
ET.SubElement(la, k("range")).text = str(int(apogee_alt_val * 1.5))
ET.SubElement(la, k("altitudeMode")).text = "absolute"

# Trajectory phases
tf = ET.SubElement(doc, k("Folder"))
ET.SubElement(tf, k("name")).text = "飛行フェーズ (Trajectory Phases)"
ET.SubElement(tf, k("open")).text = "1"

def get_phase(tv):
    if tv <= burn_time:
        return "powered"
    elif tv <= apogee_time:
        return "coast_up"
    elif tv <= apogee_time + (flight.t_final - apogee_time) * 0.7:
        return "coast_down"
    return "terminal"

phase_info = {
    "powered":    ("Phase 1: 推進飛行 (Powered Flight, 143kN)", "s_powered"),
    "coast_up":   ("Phase 2: 慣性上昇 (Coast to Apogee)", "s_coast_up"),
    "coast_down": ("Phase 3: 自由落下 (Free Fall Descent)", "s_coast_down"),
    "terminal":   ("Phase 4: 終端降下 (Terminal Descent → 太平洋着水)", "s_terminal"),
}

phases_data = {"powered": [], "coast_up": [], "coast_down": [], "terminal": []}
for pt in trajectory:
    phases_data[get_phase(pt["t"])].append(pt)

po = ["powered", "coast_up", "coast_down", "terminal"]
for i in range(len(po) - 1):
    if phases_data[po[i+1]]:
        phases_data[po[i]].append(phases_data[po[i+1]][0])

for pn in po:
    pts = phases_data[pn]
    if not pts:
        continue
    label, style_id = phase_info[pn]
    pm = ET.SubElement(tf, k("Placemark"))
    ET.SubElement(pm, k("name")).text = label
    ET.SubElement(pm, k("styleUrl")).text = f"#{style_id}"
    t0, t1 = pts[0]["t"], pts[-1]["t"]
    mx_alt = max((p["alt"] - USC_ELEV)/1000 for p in pts)
    mx_spd = max(p["speed"] for p in pts)
    ET.SubElement(pm, k("description")).text = (
        f"Time: T+{t0:.0f} - T+{t1:.0f} s\n"
        f"Max altitude: {mx_alt:.1f} km\n"
        f"Max speed: {mx_spd:.0f} m/s (Mach {mx_spd/343:.1f})"
    )
    ls_el = ET.SubElement(pm, k("LineString"))
    ET.SubElement(ls_el, k("extrude")).text = "1"
    ET.SubElement(ls_el, k("tessellate")).text = "1"
    ET.SubElement(ls_el, k("altitudeMode")).text = "absolute"
    ET.SubElement(ls_el, k("coordinates")).text = " ".join(
        f"{p['lon']:.8f},{p['lat']:.8f},{p['alt']:.1f}" for p in pts
    )

# Key events
ef = ET.SubElement(doc, k("Folder"))
ET.SubElement(ef, k("name")).text = "主要イベント (Key Events)"
ET.SubElement(ef, k("open")).text = "1"

# Launch
pm = ET.SubElement(ef, k("Placemark"))
ET.SubElement(pm, k("name")).text = "内之浦宇宙空間観測所"
ET.SubElement(pm, k("styleUrl")).text = "#i_launch"
ET.SubElement(pm, k("description")).text = (
    f"Uchinoura Space Center (USC)\n"
    f"JAXA / ISAS\n"
    f"{USC_LAT}°N, {USC_LON}°E, {USC_ELEV}m ASL\n"
    f"日本の観測ロケット発射場\n"
    f"S-520 series launched here since 1980"
)
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, k("coordinates")).text = f"{USC_LON},{USC_LAT},{USC_ELEV}"

# Burnout
bo_pt = min(trajectory, key=lambda p: abs(p["t"] - burn_time))
bo_alt_km = (bo_pt["alt"] - USC_ELEV) / 1000
pm = ET.SubElement(ef, k("Placemark"))
ET.SubElement(pm, k("name")).text = f"燃焼終了 (Burnout): {bo_alt_km:.1f} km"
ET.SubElement(pm, k("styleUrl")).text = "#i_burnout"
ET.SubElement(pm, k("description")).text = (
    f"Motor burnout at T+{burn_time:.1f} s\n"
    f"Altitude: {bo_alt_km:.1f} km\n"
    f"Speed: {bo_pt['speed']:.0f} m/s"
)
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, k("coordinates")).text = f"{bo_pt['lon']:.8f},{bo_pt['lat']:.8f},{bo_pt['alt']:.1f}"

# Apogee
ap_pt = min(trajectory, key=lambda p: abs(p["t"] - apogee_time))
pm = ET.SubElement(ef, k("Placemark"))
ET.SubElement(pm, k("name")).text = f"最高到達点: {apogee_agl/1000:.1f} km"
ET.SubElement(pm, k("styleUrl")).text = "#i_apogee"
ET.SubElement(pm, k("description")).text = (
    f"Apogee at T+{apogee_time:.0f} s\n"
    f"Altitude: {apogee_agl/1000:.1f} km AGL ({apogee_alt_val/1000:.1f} km ASL)\n"
    f"カーマンライン (100km) {'超え ✓' if apogee_agl > 100000 else '未到達'}\n"
    f"微小重力環境: ~{(flight.t_final - 2*burn_time)/60:.1f} min"
)
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "absolute"
ET.SubElement(pt_el, k("coordinates")).text = f"{ap_pt['lon']:.8f},{ap_pt['lat']:.8f},{apogee_alt_val:.1f}"

# Impact
if flight.x_impact is not None:
    ld_pt = trajectory[-1]
    dist_km = math.sqrt(flight.x_impact**2 + flight.y_impact**2) / 1000
    pm = ET.SubElement(ef, k("Placemark"))
    ET.SubElement(pm, k("name")).text = f"着水点: {dist_km:.0f} km downrange"
    ET.SubElement(pm, k("styleUrl")).text = "#i_impact"
    ET.SubElement(pm, k("description")).text = (
        f"Pacific Ocean splashdown\n"
        f"T+{flight.t_final:.0f} s\n"
        f"Distance: {dist_km:.1f} km from launch site\n"
        f"Impact velocity: {abs(flight.impact_velocity):.0f} m/s\n"
        f"太平洋上に着水 — 回収なし (expendable)"
    )
    pt_el = ET.SubElement(pm, k("Point"))
    ET.SubElement(pt_el, k("altitudeMode")).text = "clampToGround"
    ET.SubElement(pt_el, k("coordinates")).text = f"{ld_pt['lon']:.8f},{ld_pt['lat']:.8f},0"

# gx:Track animation
af = ET.SubElement(doc, k("Folder"))
ET.SubElement(af, k("name")).text = "アニメーション (Time Animation)"
ET.SubElement(af, k("open")).text = "0"
pm = ET.SubElement(af, k("Placemark"))
ET.SubElement(pm, k("name")).text = "S-520 Position"
st = ET.SubElement(pm, k("Style"))
ics = ET.SubElement(st, k("IconStyle"))
ET.SubElement(ics, k("scale")).text = "1.3"
ic = ET.SubElement(ics, k("Icon"))
ET.SubElement(ic, k("href")).text = "https://maps.google.com/mapfiles/kml/shapes/rocket.png"

track = ET.SubElement(pm, g("Track"))
ET.SubElement(track, k("altitudeMode")).text = "absolute"
base_dt = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 1, 0, 0)

anim_step = max(1.0, flight.t_final / 300)
anim_t = np.arange(0, flight.t_final + anim_step, anim_step)
if anim_t[-1] > flight.t_final:
    anim_t[-1] = flight.t_final

for tv in anim_t:
    when = base_dt + datetime.timedelta(seconds=float(tv))
    ET.SubElement(track, k("when")).text = when.strftime("%Y-%m-%dT%H:%M:%SZ")
for tv in anim_t:
    x = float(flight.x(tv))
    y = float(flight.y(tv))
    alt_v = float(flight.altitude(tv))
    lat, lon = meters_to_latlon(x, y, USC_LAT, USC_LON)
    ET.SubElement(track, g("coord")).text = f"{lon:.8f} {lat:.8f} {alt_v:.1f}"

# Flight summary
sf = ET.SubElement(doc, k("Folder"))
ET.SubElement(sf, k("name")).text = "飛行データ (Flight Summary)"
pm = ET.SubElement(sf, k("Placemark"))
ET.SubElement(pm, k("name")).text = "S-520 Flight Data Sheet"
dist_val = math.sqrt(flight.x_impact**2 + flight.y_impact**2)/1000 if flight.x_impact else 0
ET.SubElement(pm, k("description")).text = f"""<![CDATA[
<h2>JAXA S-520 観測ロケット</h2>
<h3>内之浦宇宙空間観測所からの打ち上げ</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">
<tr style="background:#1a3366;color:white;"><td colspan="2"><b>Launch Site</b></td></tr>
<tr><td>Site</td><td>内之浦宇宙空間観測所 (Uchinoura Space Center)</td></tr>
<tr><td>Organization</td><td>JAXA / ISAS</td></tr>
<tr><td>Coordinates</td><td>{USC_LAT}°N, {USC_LON}°E</td></tr>
<tr><td>Date</td><td>{tomorrow} 10:00 JST</td></tr>
<tr style="background:#1a3366;color:white;"><td colspan="2"><b>Vehicle: S-520</b></td></tr>
<tr><td>Length</td><td>9.0 m</td></tr>
<tr><td>Diameter</td><td>520 mm</td></tr>
<tr><td>Launch Mass</td><td>~2,300 kg</td></tr>
<tr><td>Motor</td><td>Single-stage solid (143 kN, 29s burn)</td></tr>
<tr><td>Total Impulse</td><td>{motor.total_impulse/1e6:.2f} MNs</td></tr>
<tr style="background:#1a3366;color:white;"><td colspan="2"><b>Flight Results (6-DOF Simulation)</b></td></tr>
<tr><td>Apogee</td><td><b>{apogee_agl/1000:.1f} km</b></td></tr>
<tr><td>Max Speed</td><td>{flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.1f})</td></tr>
<tr><td>Max Acceleration</td><td>{flight.max_acceleration:.0f} m/s² ({flight.max_acceleration/9.81:.1f} G)</td></tr>
<tr><td>Burnout Altitude</td><td>{bo_alt_km:.1f} km</td></tr>
<tr><td>Flight Time</td><td>{flight.t_final:.0f} s ({flight.t_final/60:.1f} min)</td></tr>
<tr><td>Impact Distance</td><td>{dist_val:.0f} km (Pacific Ocean)</td></tr>
<tr><td>Karman Line (100km)</td><td>{'REACHED ✓' if apogee_agl > 100000 else 'Not reached'}</td></tr>
</table>
<p><i>6-DOF Simulation by RocketPy / Standard Atmosphere / No wind</i></p>
]]>"""
pt_el = ET.SubElement(pm, k("Point"))
ET.SubElement(pt_el, k("altitudeMode")).text = "clampToGround"
ET.SubElement(pt_el, k("coordinates")).text = f"{USC_LON + 0.01},{USC_LAT + 0.01},{USC_ELEV}"

# Write
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
output_kml = "jaxa_s520_uchinoura.kml"
tree.write(output_kml, xml_declaration=True, encoding="UTF-8")

print(f"\nKML saved: {output_kml}")
print("Google Earth で開いてください！")
print(f"\n=== 参考: 実機 S-520 との比較 ===")
print(f"  実機 S-520 apogee: 300-430 km (JAXA公式)")
print(f"  シミュレーション:  {apogee_agl/1000:.1f} km")
print(f"  ※ RocketPy は大気圏内 6-DOF に最適化。高高度での")
print(f"    希薄大気モデルは簡易的なため、実機との差異あり。")
