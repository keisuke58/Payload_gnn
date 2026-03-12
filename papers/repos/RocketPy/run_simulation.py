"""
Rocket Launch Trajectory Simulation using RocketPy
- 6-DOF trajectory simulation
- Trajectory plot, altitude profile, velocity profile output
"""
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rocketpy import Environment, Flight, Rocket, SolidMotor

# ============================================================
# 1. Environment (Spaceport America, NM)
# ============================================================
env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))

# Use standard atmosphere (no network needed)
env.set_atmospheric_model(type="standard_atmosphere")

print("=== Environment ===")
print(f"  Date: {tomorrow}")
print(f"  Location: 32.99N, -106.97W, 1400m ASL")

# ============================================================
# 2. Solid Motor (Cesaroni M1670)
# ============================================================
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

print("\n=== Motor ===")
print(f"  Total impulse: {Pro75M1670.total_impulse:.1f} Ns")
print(f"  Burn time: {Pro75M1670.burn_out_time:.2f} s")
print(f"  Average thrust: {Pro75M1670.average_thrust:.1f} N")

# ============================================================
# 3. Rocket (Calisto)
# ============================================================
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

nose_cone = calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)

fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0,
    airfoil=("data/airfoils/NACA0012-radians.txt", "radians"),
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)

# Parachutes
Main = calisto.add_parachute(
    "Main", cd_s=10.0, trigger=800, sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)
Drogue = calisto.add_parachute(
    "Drogue", cd_s=1.0, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5),
)

print("\n=== Rocket: Calisto ===")
print(f"  Mass (without motor): {calisto.mass:.3f} kg")
print(f"  Radius: {calisto.radius:.4f} m")

# ============================================================
# 4. Flight Simulation
# ============================================================
print("\n=== Running Flight Simulation ===")

test_flight = Flight(
    rocket=calisto,
    environment=env,
    rail_length=5.2,
    inclination=85,  # 5 deg from vertical
    heading=0,       # North
)

print(f"  Apogee AGL: {test_flight.apogee - env.elevation:.1f} m")
print(f"  Apogee time: {test_flight.apogee_time:.2f} s")
print(f"  Max speed: {test_flight.max_speed:.1f} m/s")
print(f"  Max Mach: {test_flight.max_mach_number:.2f}")
print(f"  Max acceleration: {test_flight.max_acceleration:.1f} m/s^2")
print(f"  Out of rail speed: {test_flight.out_of_rail_velocity:.1f} m/s")
print(f"  Impact velocity: {test_flight.impact_velocity:.1f} m/s")
print(f"  Flight time: {test_flight.t_final:.1f} s")
print(f"  Landing point: ({test_flight.x_impact:.1f}, {test_flight.y_impact:.1f}) m from launch")

# ============================================================
# 5. Plot Results
# ============================================================

# --- Altitude vs Time ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

t = np.linspace(0, test_flight.t_final, 1000)

# (a) Altitude
ax = axes[0, 0]
alt = [test_flight.altitude(ti) - env.elevation for ti in t]
ax.plot(t, alt, "b-", linewidth=1.5)
ax.axhline(y=test_flight.apogee - env.elevation, color="r", linestyle="--", alpha=0.5,
           label=f"Apogee: {test_flight.apogee - env.elevation:.0f} m")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude AGL (m)")
ax.set_title("(a) Altitude Profile")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Velocity
ax = axes[0, 1]
speed = [test_flight.speed(ti) for ti in t]
ax.plot(t, speed, "r-", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("(b) Speed Profile")
ax.grid(True, alpha=0.3)

# (c) 2D Trajectory (x-z plane)
ax = axes[1, 0]
x_pos = [test_flight.x(ti) for ti in t]
z_pos = [test_flight.altitude(ti) - env.elevation for ti in t]
ax.plot(x_pos, z_pos, "g-", linewidth=1.5)
ax.set_xlabel("Downrange X (m)")
ax.set_ylabel("Altitude AGL (m)")
ax.set_title("(c) Trajectory (X-Z Plane)")
ax.grid(True, alpha=0.3)

# (d) Ground Track (x-y plane)
ax = axes[1, 1]
y_pos = [test_flight.y(ti) for ti in t]
ax.plot(x_pos, y_pos, "m-", linewidth=1.5)
ax.plot(0, 0, "r^", markersize=10, label="Launch")
ax.plot(test_flight.x_impact, test_flight.y_impact, "kx", markersize=10, label="Landing")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("(d) Ground Track")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.suptitle("Rocket Launch Simulation — Calisto (6-DOF)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("launch_simulation_results.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: launch_simulation_results.png")

# --- 3D Trajectory ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_pos, y_pos, z_pos, "b-", linewidth=1.5)
ax.plot([0], [0], [0], "r^", markersize=10, label="Launch")
ax.plot([test_flight.x_impact], [test_flight.y_impact], [0], "kx", markersize=10, label="Landing")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Altitude AGL (m)")
ax.set_title("3D Trajectory — Calisto Rocket")
ax.legend()
plt.savefig("launch_simulation_3d.png", dpi=150, bbox_inches="tight")
print("Figure saved: launch_simulation_3d.png")

# Export KML for Google Earth
test_flight.export_kml(
    file_name="trajectory.kml",
    extrude=True,
    altitude_mode="relative_to_ground",
)
print("KML saved: trajectory.kml")

print("\nSimulation complete!")
