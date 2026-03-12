"""
Generate a professional KML for Google Earth from RocketPy simulation.
Features:
  - Color-coded trajectory phases (Powered / Coast / Drogue / Main)
  - Launch, Apogee, Landing placemarks with details
  - Altitude-colored path with extrusion
  - gx:Track time animation
  - Camera/LookAt for flyover view
"""
import datetime
import math
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rocketpy import Environment, Flight, Rocket, SolidMotor

# ============================================================
# 1. Run Simulation (same as before)
# ============================================================
env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
env.set_atmospheric_model(type="standard_atmosphere")

Pro75M1670 = SolidMotor(
    thrust_source="data/motors/cesaroni/Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
    power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)
calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.618,
    angular_position=45,
)
calisto.add_motor(Pro75M1670, position=-1.255)
calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)
calisto.add_trapezoidal_fins(
    n=4, root_chord=0.120, tip_chord=0.060, span=0.110,
    position=-1.04956, cant_angle=0,
    airfoil=("data/airfoils/NACA0012-radians.txt", "radians"),
)
calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)
calisto.add_parachute(
    "Main", cd_s=10.0, trigger=800, sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)
calisto.add_parachute(
    "Drogue", cd_s=1.0, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)

print("Running simulation...")
flight = Flight(
    rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0,
)

# ============================================================
# 2. Extract trajectory data
# ============================================================
lat0, lon0, elev0 = 32.990254, -106.974998, 1400.0

# Earth radius for meter → degree conversion
R_earth = 6371000.0

def meters_to_latlon(x, y, lat0, lon0):
    """Convert local x,y (meters) to lat,lon."""
    dlat = y / R_earth * (180.0 / math.pi)
    dlon = x / (R_earth * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon

# Sample trajectory at high resolution
dt = 0.5  # seconds
t_values = np.arange(0, flight.t_final + dt, dt)
if t_values[-1] > flight.t_final:
    t_values[-1] = flight.t_final

trajectory = []
for t in t_values:
    x = float(flight.x(t))
    y = float(flight.y(t))
    alt = float(flight.altitude(t))
    speed = float(flight.speed(t))
    lat, lon = meters_to_latlon(x, y, lat0, lon0)
    trajectory.append({
        "t": float(t), "lat": lat, "lon": lon, "alt": alt, "speed": speed,
        "x": x, "y": y,
    })

# Key events
burn_time = float(Pro75M1670.burn_out_time)
apogee_time = float(flight.apogee_time)
apogee_alt = float(flight.apogee)

# Find drogue and main deployment times from parachute events
drogue_time = apogee_time  # drogue at apogee
# Main at 800m AGL
main_deploy_alt = elev0 + 800.0
main_time = None
for i in range(len(trajectory) - 1):
    if trajectory[i]["t"] > apogee_time and trajectory[i]["alt"] > main_deploy_alt and trajectory[i+1]["alt"] <= main_deploy_alt:
        main_time = trajectory[i+1]["t"]
        break
if main_time is None:
    main_time = flight.t_final * 0.8

print(f"  Burn time: {burn_time:.1f} s")
print(f"  Apogee: {apogee_alt:.0f} m ASL at {apogee_time:.1f} s")
print(f"  Drogue deploy: {drogue_time:.1f} s")
print(f"  Main deploy: ~{main_time:.1f} s")
print(f"  Flight time: {flight.t_final:.1f} s")

# ============================================================
# 3. Build KML
# ============================================================
KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"

ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)

def kml_tag(tag):
    return f"{{{KML_NS}}}{tag}"

def gx_tag(tag):
    return f"{{{GX_NS}}}{tag}"

root = ET.Element(kml_tag("kml"))
doc = ET.SubElement(root, kml_tag("Document"))
ET.SubElement(doc, kml_tag("name")).text = "Calisto Rocket — Launch Simulation"
ET.SubElement(doc, kml_tag("open")).text = "1"
desc = ET.SubElement(doc, kml_tag("description"))
desc.text = (
    f"6-DOF Trajectory Simulation (RocketPy)\n"
    f"Date: {tomorrow}\n"
    f"Location: Spaceport America, NM\n"
    f"Apogee: {apogee_alt - elev0:.0f} m AGL ({apogee_alt:.0f} m ASL)\n"
    f"Max Speed: {flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})\n"
    f"Flight Time: {flight.t_final:.0f} s"
)

# --- Styles ---
phase_styles = {
    "powered":  {"color": "ff0000ff", "width": "4"},  # Red (BBGGRR in KML)
    "coast":    {"color": "ff00a5ff", "width": "3"},   # Orange
    "drogue":   {"color": "ffff9900", "width": "2.5"}, # Blue-ish
    "main":     {"color": "ff00ff00", "width": "2"},    # Green
}

for name, style_def in phase_styles.items():
    style = ET.SubElement(doc, kml_tag("Style"), id=f"style_{name}")
    ls = ET.SubElement(style, kml_tag("LineStyle"))
    ET.SubElement(ls, kml_tag("color")).text = style_def["color"]
    ET.SubElement(ls, kml_tag("width")).text = style_def["width"]
    ps = ET.SubElement(style, kml_tag("PolyStyle"))
    ET.SubElement(ps, kml_tag("color")).text = style_def["color"][:2] + "ffffff"
    ET.SubElement(ps, kml_tag("fill")).text = "1"

# Placemark styles
for pm_name, icon_href, color in [
    ("launch", "https://maps.google.com/mapfiles/kml/shapes/triangle.png", "ff0000ff"),
    ("apogee", "https://maps.google.com/mapfiles/kml/shapes/star.png", "ff00ffff"),
    ("landing", "https://maps.google.com/mapfiles/kml/shapes/cross-hairs.png", "ff00ff00"),
]:
    style = ET.SubElement(doc, kml_tag("Style"), id=f"style_{pm_name}")
    is_elem = ET.SubElement(style, kml_tag("IconStyle"))
    ET.SubElement(is_elem, kml_tag("color")).text = color
    ET.SubElement(is_elem, kml_tag("scale")).text = "1.4"
    icon = ET.SubElement(is_elem, kml_tag("Icon"))
    ET.SubElement(icon, kml_tag("href")).text = icon_href
    ls = ET.SubElement(style, kml_tag("LabelStyle"))
    ET.SubElement(ls, kml_tag("scale")).text = "1.2"

# --- LookAt (initial camera) ---
lookat = ET.SubElement(doc, kml_tag("LookAt"))
ET.SubElement(lookat, kml_tag("longitude")).text = str(lon0)
ET.SubElement(lookat, kml_tag("latitude")).text = str(lat0)
ET.SubElement(lookat, kml_tag("altitude")).text = str(apogee_alt / 2)
ET.SubElement(lookat, kml_tag("heading")).text = "0"
ET.SubElement(lookat, kml_tag("tilt")).text = "60"
ET.SubElement(lookat, kml_tag("range")).text = str(int(apogee_alt * 1.5))
ET.SubElement(lookat, kml_tag("altitudeMode")).text = "absolute"

# --- Trajectory Phases Folder ---
traj_folder = ET.SubElement(doc, kml_tag("Folder"))
ET.SubElement(traj_folder, kml_tag("name")).text = "Trajectory Phases"
ET.SubElement(traj_folder, kml_tag("open")).text = "1"

def get_phase(t):
    if t <= burn_time:
        return "powered"
    elif t <= drogue_time:
        return "coast"
    elif t <= main_time:
        return "drogue"
    else:
        return "main"

phase_labels = {
    "powered": "Phase 1: Powered Flight (Motor Burn)",
    "coast": "Phase 2: Coast (Ballistic Arc)",
    "drogue": "Phase 3: Drogue Descent",
    "main": "Phase 4: Main Parachute Descent",
}

# Group trajectory points by phase
phases_data = {"powered": [], "coast": [], "drogue": [], "main": []}
for pt in trajectory:
    phase = get_phase(pt["t"])
    phases_data[phase].append(pt)

# Ensure continuity: add first point of next phase to end of current phase
phase_order = ["powered", "coast", "drogue", "main"]
for i in range(len(phase_order) - 1):
    curr = phase_order[i]
    nxt = phase_order[i + 1]
    if phases_data[nxt]:
        phases_data[curr].append(phases_data[nxt][0])

for phase_name in phase_order:
    pts = phases_data[phase_name]
    if not pts:
        continue

    pm = ET.SubElement(traj_folder, kml_tag("Placemark"))
    ET.SubElement(pm, kml_tag("name")).text = phase_labels[phase_name]
    ET.SubElement(pm, kml_tag("styleUrl")).text = f"#style_{phase_name}"

    t_start = pts[0]["t"]
    t_end = pts[-1]["t"]
    max_alt = max(p["alt"] - elev0 for p in pts)
    max_spd = max(p["speed"] for p in pts)
    ET.SubElement(pm, kml_tag("description")).text = (
        f"Time: {t_start:.1f} - {t_end:.1f} s\n"
        f"Max altitude AGL: {max_alt:.0f} m\n"
        f"Max speed: {max_spd:.0f} m/s"
    )

    ls = ET.SubElement(pm, kml_tag("LineString"))
    ET.SubElement(ls, kml_tag("extrude")).text = "1"
    ET.SubElement(ls, kml_tag("tessellate")).text = "1"
    ET.SubElement(ls, kml_tag("altitudeMode")).text = "absolute"

    coords_str = " ".join(
        f"{p['lon']:.8f},{p['lat']:.8f},{p['alt']:.1f}" for p in pts
    )
    ET.SubElement(ls, kml_tag("coordinates")).text = coords_str

# --- Key Placemarks Folder ---
pm_folder = ET.SubElement(doc, kml_tag("Folder"))
ET.SubElement(pm_folder, kml_tag("name")).text = "Key Events"
ET.SubElement(pm_folder, kml_tag("open")).text = "1"

# Launch
pm = ET.SubElement(pm_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = "Launch Site"
ET.SubElement(pm, kml_tag("styleUrl")).text = "#style_launch"
ET.SubElement(pm, kml_tag("description")).text = (
    f"Spaceport America, NM\n"
    f"Lat: {lat0:.6f}, Lon: {lon0:.6f}\n"
    f"Elevation: {elev0:.0f} m ASL\n"
    f"Rail length: 5.2 m\n"
    f"Inclination: 85 deg\n"
    f"Out-of-rail speed: {flight.out_of_rail_velocity:.1f} m/s"
)
pt = ET.SubElement(pm, kml_tag("Point"))
ET.SubElement(pt, kml_tag("altitudeMode")).text = "absolute"
ET.SubElement(pt, kml_tag("coordinates")).text = f"{lon0},{lat0},{elev0}"

# Apogee
apogee_pt = min(trajectory, key=lambda p: abs(p["t"] - apogee_time))
pm = ET.SubElement(pm_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = f"Apogee: {apogee_alt - elev0:.0f} m AGL"
ET.SubElement(pm, kml_tag("styleUrl")).text = "#style_apogee"
ET.SubElement(pm, kml_tag("description")).text = (
    f"Altitude: {apogee_alt:.0f} m ASL ({apogee_alt - elev0:.0f} m AGL)\n"
    f"Time: {apogee_time:.1f} s\n"
    f"Drogue parachute deployed here"
)
pt = ET.SubElement(pm, kml_tag("Point"))
ET.SubElement(pt, kml_tag("altitudeMode")).text = "absolute"
ET.SubElement(pt, kml_tag("coordinates")).text = f"{apogee_pt['lon']:.8f},{apogee_pt['lat']:.8f},{apogee_alt:.1f}"

# Main deploy
main_pt = min(trajectory, key=lambda p: abs(p["t"] - main_time))
pm = ET.SubElement(pm_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = "Main Parachute Deploy (800m AGL)"
ET.SubElement(pm, kml_tag("styleUrl")).text = "#style_landing"
ET.SubElement(pm, kml_tag("description")).text = (
    f"Altitude: ~{main_deploy_alt:.0f} m ASL (800 m AGL)\n"
    f"Time: ~{main_time:.1f} s\n"
    f"Main parachute (Cd*S=10.0) deployed"
)
pt = ET.SubElement(pm, kml_tag("Point"))
ET.SubElement(pt, kml_tag("altitudeMode")).text = "absolute"
ET.SubElement(pt, kml_tag("coordinates")).text = f"{main_pt['lon']:.8f},{main_pt['lat']:.8f},{main_pt['alt']:.1f}"

# Landing
land_pt = trajectory[-1]
pm = ET.SubElement(pm_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = "Landing"
ET.SubElement(pm, kml_tag("styleUrl")).text = "#style_landing"
ET.SubElement(pm, kml_tag("description")).text = (
    f"Impact velocity: {flight.impact_velocity:.1f} m/s\n"
    f"Distance from launch: {math.sqrt(flight.x_impact**2 + flight.y_impact**2):.0f} m\n"
    f"Offset: ({flight.x_impact:.1f}, {flight.y_impact:.1f}) m\n"
    f"Time: {flight.t_final:.1f} s"
)
pt = ET.SubElement(pm, kml_tag("Point"))
ET.SubElement(pt, kml_tag("altitudeMode")).text = "clampToGround"
ET.SubElement(pt, kml_tag("coordinates")).text = f"{land_pt['lon']:.8f},{land_pt['lat']:.8f},{elev0}"

# --- gx:Track (Time Animation) Folder ---
track_folder = ET.SubElement(doc, kml_tag("Folder"))
ET.SubElement(track_folder, kml_tag("name")).text = "Animated Track (Time Slider)"
ET.SubElement(track_folder, kml_tag("open")).text = "0"

pm = ET.SubElement(track_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = "Rocket Position"

# Style for track icon
style = ET.SubElement(pm, kml_tag("Style"))
is_elem = ET.SubElement(style, kml_tag("IconStyle"))
ET.SubElement(is_elem, kml_tag("scale")).text = "1.2"
icon = ET.SubElement(is_elem, kml_tag("Icon"))
ET.SubElement(icon, kml_tag("href")).text = "https://maps.google.com/mapfiles/kml/shapes/rocket.png"

track = ET.SubElement(pm, gx_tag("Track"))
ET.SubElement(track, kml_tag("altitudeMode")).text = "absolute"

# Use launch time as base
base_dt = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 12, 0, 0)

# Sample at lower rate for animation (every 1s)
anim_dt = 1.0
anim_times = np.arange(0, flight.t_final + anim_dt, anim_dt)
if anim_times[-1] > flight.t_final:
    anim_times[-1] = flight.t_final

for t in anim_times:
    when = base_dt + datetime.timedelta(seconds=float(t))
    ET.SubElement(track, kml_tag("when")).text = when.strftime("%Y-%m-%dT%H:%M:%SZ")

for t in anim_times:
    x = float(flight.x(t))
    y = float(flight.y(t))
    alt = float(flight.altitude(t))
    lat, lon = meters_to_latlon(x, y, lat0, lon0)
    coord = ET.SubElement(track, gx_tag("coord"))
    coord.text = f"{lon:.8f} {lat:.8f} {alt:.1f}"

# --- Flight Info Folder ---
info_folder = ET.SubElement(doc, kml_tag("Folder"))
ET.SubElement(info_folder, kml_tag("name")).text = "Flight Summary"
ET.SubElement(info_folder, kml_tag("open")).text = "0"

pm = ET.SubElement(info_folder, kml_tag("Placemark"))
ET.SubElement(pm, kml_tag("name")).text = "Flight Data"
ET.SubElement(pm, kml_tag("description")).text = f"""<![CDATA[
<h2>Calisto Rocket — Flight Summary</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><td><b>Motor</b></td><td>Cesaroni M1670 (Pro75)</td></tr>
<tr><td><b>Total Impulse</b></td><td>{Pro75M1670.total_impulse:.0f} Ns</td></tr>
<tr><td><b>Burn Time</b></td><td>{burn_time:.2f} s</td></tr>
<tr><td><b>Rocket Mass</b></td><td>{calisto.mass:.1f} kg (without motor)</td></tr>
<tr><td><b>Apogee (AGL)</b></td><td>{apogee_alt - elev0:.0f} m</td></tr>
<tr><td><b>Apogee (ASL)</b></td><td>{apogee_alt:.0f} m</td></tr>
<tr><td><b>Max Speed</b></td><td>{flight.max_speed:.0f} m/s (Mach {flight.max_mach_number:.2f})</td></tr>
<tr><td><b>Max Acceleration</b></td><td>{flight.max_acceleration:.0f} m/s² ({flight.max_acceleration/9.81:.1f} G)</td></tr>
<tr><td><b>Out-of-Rail Speed</b></td><td>{flight.out_of_rail_velocity:.1f} m/s</td></tr>
<tr><td><b>Impact Velocity</b></td><td>{abs(flight.impact_velocity):.1f} m/s</td></tr>
<tr><td><b>Flight Time</b></td><td>{flight.t_final:.0f} s</td></tr>
<tr><td><b>Landing Distance</b></td><td>{math.sqrt(flight.x_impact**2 + flight.y_impact**2):.0f} m</td></tr>
</table>
<p><i>Simulation: RocketPy 6-DOF, Standard Atmosphere</i></p>
]]>"""
pt = ET.SubElement(pm, kml_tag("Point"))
ET.SubElement(pt, kml_tag("altitudeMode")).text = "clampToGround"
ET.SubElement(pt, kml_tag("coordinates")).text = f"{lon0 + 0.002},{lat0 + 0.002},{elev0}"

# ============================================================
# 4. Write KML
# ============================================================
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")

output_path = "trajectory_pro.kml"
tree.write(output_path, xml_declaration=True, encoding="UTF-8")
print(f"\nKML saved: {output_path}")
print("Open in Google Earth Pro or Google Earth Web to view!")
