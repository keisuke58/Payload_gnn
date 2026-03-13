#!/usr/bin/env python3
"""
H3 Virtual Twin — 6DOF Flight Simulator (Phase C)

Quaternion-based 6DOF trajectory simulation for multi-stage orbital rockets.
Integrates Phase A modules (propulsion, aerodynamics, aerothermal).

Features:
  - Quaternion kinematics (gimbal-lock free)
  - WGS84 ellipsoid + J2 gravity perturbation
  - Rotating Earth (Coriolis & centrifugal)
  - Event-driven staging (SRB sep, fairing sep, MECO, stage sep, SECO)
  - Lat/Lon/Alt output for CesiumJS visualization

References:
  - Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics"
  - Wertz, "Space Mission Engineering: The New SMAD"
  - JAXA H3 Rocket User's Manual
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Callable
import math

from .propulsion import H3PropulsionSystem, G0
from .aerodynamics import H3Aerodynamics, atmosphere_isa

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

# WGS84 Ellipsoid
R_EARTH = 6_378_137.0          # Equatorial radius [m]
R_POLAR = 6_356_752.314245     # Polar radius [m]
F_EARTH = 1 / 298.257223563    # Flattening
E2 = 2 * F_EARTH - F_EARTH**2  # Eccentricity squared
MU_EARTH = 3.986004418e14      # Gravitational parameter [m³/s²]
J2 = 1.08263e-3                # J2 zonal harmonic
OMEGA_EARTH = 7.2921159e-5     # Earth rotation rate [rad/s]

# Tanegashima Space Center (Yoshinobu Launch Complex)
LAT_TNSC = 30.4009             # [deg]
LON_TNSC = 130.9751            # [deg]
ALT_TNSC = 20.0                # [m] elevation
LAUNCH_AZIMUTH = 120.0         # [deg] typical SSO azimuth (southeast)


# ══════════════════════════════════════════════════════════════
# Coordinate transforms
# ══════════════════════════════════════════════════════════════

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt: float):
    """Geodetic (WGS84) → ECEF [m]."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    N = R_EARTH / np.sqrt(1 - E2 * sl**2)
    x = (N + alt) * cl * np.cos(lon)
    y = (N + alt) * cl * np.sin(lon)
    z = (N * (1 - E2) + alt) * sl
    return np.array([x, y, z])


def ecef_to_geodetic(r_ecef):
    """ECEF → Geodetic (WGS84) using Bowring iterative method."""
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    # Initial guess
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(5):
        sl = np.sin(lat)
        N = R_EARTH / np.sqrt(1 - E2 * sl**2)
        lat = np.arctan2(z + E2 * N * sl, p)
    sl = np.sin(lat)
    N = R_EARTH / np.sqrt(1 - E2 * sl**2)
    alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) / abs(sl) - N * (1 - E2)
    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float):
    """Rotation matrix: ECEF → ENU (East-North-Up)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)
    return np.array([
        [-sn,       cn,      0  ],
        [-sl * cn, -sl * sn,  cl],
        [ cl * cn,  cl * sn,  sl],
    ])


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2. Convention: q = [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q: v' = q ⊗ v ⊗ q*."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    r = quat_multiply(quat_multiply(q, qv), qc)
    return r[1:]


def quat_from_euler(roll_deg, pitch_deg, yaw_deg):
    """ZYX Euler angles → quaternion [w, x, y, z]."""
    r = np.radians(roll_deg) / 2
    p = np.radians(pitch_deg) / 2
    y = np.radians(yaw_deg) / 2
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_to_euler(q):
    """Quaternion → ZYX Euler angles [roll, pitch, yaw] in degrees."""
    w, x, y, z = q
    # Roll
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr, cosr)
    # Pitch
    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw
    siny = 2 * (w*z + x*y)
    cosy = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def quat_normalize(q):
    """Normalize quaternion."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════
# Gravity model (J2)
# ══════════════════════════════════════════════════════════════

def gravity_j2(r_ecef):
    """J2 gravity acceleration in ECEF frame [m/s²]."""
    x, y, z = r_ecef
    r = np.linalg.norm(r_ecef)
    if r < R_EARTH * 0.9:
        return np.array([0.0, 0.0, -G0])

    r2 = r * r
    r5 = r**5
    z2_r2 = z**2 / r2
    coeff = -MU_EARTH / r**3
    j2_term = 1.5 * J2 * (R_EARTH / r)**2

    ax = coeff * x * (1 + j2_term * (1 - 5 * z2_r2))
    ay = coeff * y * (1 + j2_term * (1 - 5 * z2_r2))
    az = coeff * z * (1 + j2_term * (3 - 5 * z2_r2))
    return np.array([ax, ay, az])


# ══════════════════════════════════════════════════════════════
# Atmosphere (wrapper for Phase A ISA model)
# ══════════════════════════════════════════════════════════════

def get_atmosphere(altitude: float) -> dict:
    """Atmosphere state at altitude [m]. Returns T, p, rho, a, mu."""
    if altitude < 0:
        altitude = 0.0
    if altitude > 300_000:
        return {"T": 250.0, "p": 0.0, "rho": 0.0, "a": 300.0, "mu": 1e-5}
    return atmosphere_isa(altitude)


# ══════════════════════════════════════════════════════════════
# Flight Event System
# ══════════════════════════════════════════════════════════════

@dataclass
class FlightEvent6DOF:
    """Flight event with time and/or condition trigger."""
    name: str
    t_nominal: float              # Nominal time [s]
    triggered: bool = False
    t_actual: float = 0.0
    condition: Optional[Callable] = None  # f(t, state_dict) → bool

    def check(self, t: float, state_dict: dict) -> bool:
        if self.triggered:
            return False
        if self.condition is not None:
            return self.condition(t, state_dict)
        return t >= self.t_nominal


# ══════════════════════════════════════════════════════════════
# 6DOF State & Telemetry
# ══════════════════════════════════════════════════════════════

@dataclass
class Telemetry6DOF:
    """Stores flight telemetry at each timestep."""
    t: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    lon: list = field(default_factory=list)
    alt: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    mach: list = field(default_factory=list)
    q_dyn: list = field(default_factory=list)
    accel_g: list = field(default_factory=list)
    mass: list = field(default_factory=list)
    thrust: list = field(default_factory=list)
    drag: list = field(default_factory=list)
    gamma: list = field(default_factory=list)       # Flight path angle [deg]
    pitch: list = field(default_factory=list)        # Pitch angle [deg]
    roll: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    alpha: list = field(default_factory=list)        # Angle of attack [deg]
    T_nose: list = field(default_factory=list)       # Nose temperature [K]
    downrange: list = field(default_factory=list)    # Ground track distance [km]
    v_inertial: list = field(default_factory=list)   # Inertial velocity [m/s]
    phase: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict of numpy arrays."""
        return {k: np.array(v) for k, v in self.__dict__.items()
                if isinstance(v, list) and len(v) > 0}


# ══════════════════════════════════════════════════════════════
# H3 6DOF Simulator
# ══════════════════════════════════════════════════════════════

class H3SixDOF:
    """
    6DOF flight simulator for H3 rocket.

    State vector (13 components):
      [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
       ECEF pos   ECEF vel    quaternion     angular vel (body)

    Integrates translational + rotational dynamics with:
      - J2 gravity
      - Rotating Earth
      - Aerodynamic forces (from Phase A tables)
      - Thrust with gimbal
      - Event-driven staging
    """

    def __init__(self, config: str = "H3-22S",
                 launch_azimuth: float = LAUNCH_AZIMUTH,
                 payload_mass: float = 4000.0,
                 dt_record: float = 0.5):
        self.config = config
        self.launch_azimuth = launch_azimuth
        self.payload_mass = payload_mass
        self.dt_record = dt_record

        # Phase A modules
        self.propulsion = H3PropulsionSystem(config)
        self.propulsion.payload_mass = payload_mass
        self.aero = H3Aerodynamics(config)

        # Mass tracking
        self._s1_prop = self.propulsion.s1_propellant
        self._s2_prop = self.propulsion.s2_propellant
        self._srb_prop = sum(b.propellant_mass for b in self.propulsion.srb_boosters)
        self._s1_prop_remaining = self._s1_prop
        self._s2_prop_remaining = self._s2_prop
        self._srb_prop_remaining = self._srb_prop

        # Current mass
        self._mass = self.propulsion.liftoff_mass

        # Stage flags
        self._srb_active = self.propulsion.n_srb > 0
        self._srb_separated = False
        self._s1_active = True
        self._s2_active = False
        self._fairing_on = True

        # Thermal state
        self._T_nose = 300.0  # [K]

        # Telemetry
        self.telemetry = Telemetry6DOF()
        self.events_log = []

        # Launch site
        self.lat0 = LAT_TNSC
        self.lon0 = LON_TNSC
        self.alt0 = ALT_TNSC

        # Dispersion factor for Cd (set by Monte Carlo)
        self._cd_dispersion = 1.0

        # Build events
        self._build_events()

        # Pitch schedule (piecewise-linear, approximating H3 optimal trajectory)
        # Format: [(t, pitch_deg), ...] — pitch above horizontal
        self._pitch_schedule = [
            (0.0,   90.0),   # Vertical
            (8.0,   90.0),   # Hold vertical
            (20.0,  78.0),   # Pitchover kick
            (60.0,  50.0),   # Through Max-Q
            (113.0, 28.0),   # SRB burnout
            (200.0, 12.0),   # Mid S1
            (290.0,  3.0),   # MECO
            (350.0,  1.0),   # S2 early
            (600.0, -0.5),   # Near-horizontal insertion
            (1000.0, -0.5),
        ]

        # Inertia (simplified cylinder model)
        self._Iyy_s1 = 2.5e7   # [kg·m²] — 1st stage + SRB
        self._Iyy_s2 = 0.5e6   # [kg·m²] — 2nd stage

    def _build_events(self):
        """Create event timeline for H3-22S."""
        self.events = {}

        if self.propulsion.n_srb > 0:
            srb_burn = self.propulsion.srb_boosters[0].burn_time
            self.events["SRB_BURNOUT"] = FlightEvent6DOF(
                "SRB_BURNOUT", srb_burn,
                condition=lambda t, s: (
                    self._srb_prop_remaining <= 500 or t >= srb_burn
                )
            )
            self.events["SRB_SEP"] = FlightEvent6DOF(
                "SRB_SEP", srb_burn + 2.0,
                condition=lambda t, s: (
                    self.events["SRB_BURNOUT"].triggered and
                    t >= self.events["SRB_BURNOUT"].t_actual + 2.0
                )
            )

        self.events["FAIRING_SEP"] = FlightEvent6DOF(
            "FAIRING_SEP", 230.0,
            condition=lambda t, s: s.get("alt", 0) > 120_000 and t > 100
        )
        self.events["MECO"] = FlightEvent6DOF(
            "MECO", 290.0,
            condition=lambda t, s: self._s1_prop_remaining <= 500
        )
        self.events["STAGE_SEP"] = FlightEvent6DOF(
            "STAGE_SEP", 295.0,
            condition=lambda t, s: (
                self.events["MECO"].triggered and
                t >= self.events["MECO"].t_actual + 3.0
            )
        )
        self.events["S2_IGNITION"] = FlightEvent6DOF(
            "S2_IGNITION", 300.0,
            condition=lambda t, s: (
                self.events["STAGE_SEP"].triggered and
                t >= self.events["STAGE_SEP"].t_actual + 5.0
            )
        )
        # SECO: propellant depletion OR target orbital velocity reached
        V_ORBIT_TARGET = 7_800.0  # [m/s] circular LEO velocity
        self.events["SECO"] = FlightEvent6DOF(
            "SECO", 900.0,
            condition=lambda t, s: (
                self._s2_active and (
                    self._s2_prop_remaining <= 0 or
                    s.get("speed", 0) >= V_ORBIT_TARGET
                )
            )
        )

    def _initial_state(self) -> np.ndarray:
        """Initial state vector at launch pad."""
        # ECEF position
        r0 = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)

        # In ECEF (rotating frame), pad velocity = 0
        v0 = np.array([0.0, 0.0, 0.0])

        # Initial attitude: rocket pointing UP in ENU → convert to ECEF
        # Body +x = rocket axis (up), body +y = east, body +z = north
        # At launch: pitch = 90° (vertical), heading = launch_azimuth
        lat_r = np.radians(self.lat0)
        lon_r = np.radians(self.lon0)
        az_r = np.radians(self.launch_azimuth)

        # ENU up direction in ECEF
        up_ecef = np.array([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r)
        ])
        # East in ECEF
        east_ecef = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
        # North in ECEF
        north_ecef = np.cross(up_ecef, east_ecef)
        north_ecef = np.array([
            -np.sin(lat_r) * np.cos(lon_r),
            -np.sin(lat_r) * np.sin(lon_r),
            np.cos(lat_r)
        ])

        # Rocket body x-axis points along thrust direction
        # At launch: straight up
        body_x = up_ecef  # thrust axis = up

        # Body y-axis: in the downrange direction (azimuth)
        downrange = np.cos(az_r) * north_ecef + np.sin(az_r) * east_ecef
        body_z = downrange
        body_y = np.cross(body_z, body_x)
        body_y /= np.linalg.norm(body_y)
        body_z = np.cross(body_x, body_y)

        # Rotation matrix body→ECEF (columns = body axes in ECEF)
        R_be = np.column_stack([body_x, body_y, body_z])

        # Convert to quaternion
        q0 = self._rotmat_to_quat(R_be)

        # Angular velocity = 0
        w0 = np.array([0.0, 0.0, 0.0])

        return np.concatenate([r0, v0, q0, w0])

    @staticmethod
    def _rotmat_to_quat(R):
        """Rotation matrix → quaternion [w, x, y, z]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 2 * np.sqrt(tr + 1)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return quat_normalize(np.array([w, x, y, z]))

    def _get_body_axes(self, q):
        """Get body x, y, z axes in ECEF from quaternion."""
        body_x = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        body_y = quat_rotate(q, np.array([0.0, 1.0, 0.0]))
        body_z = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
        return body_x, body_y, body_z

    def _compute_forces(self, t, state):
        """
        Compute total force and torque in ECEF frame.

        Returns:
            f_total: [fx, fy, fz] in ECEF [N]
            tau_total: [tx, ty, tz] in body frame [N·m]
            info: dict with derived quantities
        """
        r = state[0:3]
        v = state[3:6]
        q = quat_normalize(state[6:10])
        w_body = state[10:13]

        # Geodetic position
        lat, lon, alt = ecef_to_geodetic(r)
        alt = max(alt, 0.0)

        # Body axes in ECEF
        body_x, body_y, body_z = self._get_body_axes(q)

        # In ECEF, atmosphere co-rotates with Earth → v_air = v_ecef
        v_air = np.linalg.norm(v)

        # Atmosphere
        atm = get_atmosphere(alt)
        rho = atm["rho"]
        a_sound = atm["a"]
        mach = v_air / a_sound if a_sound > 1.0 else 0.0

        # Dynamic pressure
        q_dyn = 0.5 * rho * v_air**2

        # Angle of attack: angle between velocity and body x-axis (thrust)
        if v_air > 1.0:
            v_hat = v / v_air
            cos_alpha = np.clip(np.dot(v_hat, body_x), -1.0, 1.0)
            alpha_rad = np.arccos(cos_alpha)
        else:
            alpha_rad = 0.0
        alpha_deg = np.degrees(alpha_rad)

        # ── Aerodynamic forces ──
        if v_air > 1.0 and rho > 1e-8:
            aero_result = self.aero.get_coefficients(mach, min(alpha_deg, 15.0))
            Cd = aero_result["Cd"] * self._cd_dispersion
            Cn = aero_result["Cn"]

            # Reference area
            A_ref = self.aero.geometry.S_ref

            # Drag (opposing velocity)
            F_drag = q_dyn * A_ref * Cd
            f_drag = -F_drag * (v / v_air)

            # Normal force (perpendicular to velocity, in plane of alpha)
            if alpha_deg > 0.01:
                # Normal direction: component of body_x perpendicular to v
                n_hat = body_x - np.dot(body_x, v / v_air) * (v / v_air)
                n_norm = np.linalg.norm(n_hat)
                if n_norm > 1e-6:
                    n_hat /= n_norm
                else:
                    n_hat = np.zeros(3)
                F_normal = q_dyn * A_ref * Cn
                f_normal = F_normal * n_hat
            else:
                F_normal = 0.0
                f_normal = np.zeros(3)
                F_drag = q_dyn * A_ref * Cd
        else:
            f_drag = np.zeros(3)
            f_normal = np.zeros(3)
            F_drag = 0.0
            F_normal = 0.0

        # ── Thrust ──
        thrust_mag = 0.0
        mdot_total = 0.0
        mdot_s1 = 0.0
        mdot_s2 = 0.0
        mdot_srb = 0.0

        if self._srb_active and not self._srb_separated:
            for b in self.propulsion.srb_boosters:
                r_srb = b.get_thrust(t, alt)
                if r_srb["active"]:
                    thrust_mag += r_srb["thrust"]
                    # Use design-consistent mdot (propellant_mass / burn_time)
                    # to avoid thrust/Isp mismatch depleting propellant too fast
                    mdot_srb += b.mdot_avg

        if self._s1_active:
            for e in self.propulsion.le9_engines:
                r_le9 = e.get_thrust(alt, throttle=1.0)
                thrust_mag += r_le9["thrust"]
                mdot_s1 += r_le9["mdot"]

        if self._s2_active and self.propulsion.le5b_engine:
            r_s2 = self.propulsion.le5b_engine.get_thrust(alt)
            thrust_mag += r_s2["thrust"]
            mdot_s2 += r_s2["mdot"]

        # Clamp propellant consumption
        if self._srb_prop_remaining <= 0:
            mdot_srb = 0.0
            # Remove SRB thrust
            if not self._srb_separated:
                thrust_mag -= sum(
                    b.get_thrust(t, alt)["thrust"]
                    for b in self.propulsion.srb_boosters
                    if b.get_thrust(t, alt)["active"]
                )
                thrust_mag = max(thrust_mag, 0.0)

        if self._s1_prop_remaining <= 0:
            mdot_s1 = 0.0
            if self._s1_active:
                # Kill LE-9 thrust
                thrust_mag -= sum(
                    e.get_thrust(alt)["thrust"]
                    for e in self.propulsion.le9_engines
                )
                thrust_mag = max(thrust_mag, 0.0)

        if self._s2_prop_remaining <= 0:
            mdot_s2 = 0.0
            if self._s2_active and self.propulsion.le5b_engine:
                thrust_mag -= self.propulsion.le5b_engine.get_thrust(alt)["thrust"]
                thrust_mag = max(thrust_mag, 0.0)

        mdot_total = mdot_s1 + mdot_s2 + mdot_srb

        # Thrust direction = body x-axis (+ small gimbal for attitude control)
        f_thrust = thrust_mag * body_x

        # ── Gravity (J2) ──
        f_grav = self._mass * gravity_j2(r)

        # ── Total force ──
        f_total = f_thrust + f_drag + f_normal + f_grav

        # ── Torques (simplified) ──
        # Aerodynamic restoring torque (weathercock stability)
        tau_aero = np.zeros(3)
        if alpha_deg > 0.01 and F_normal > 0:
            # Restoring moment about CoM (body y-axis)
            # Positive Cn → nose into wind → stabilizing
            arm = 5.0  # [m] rough CP-CG distance
            tau_aero[1] = -F_normal * arm * np.sin(alpha_rad)

        # Pitch damping
        tau_damp = -0.1 * self._get_Iyy() * w_body

        tau_total = tau_aero + tau_damp

        # ── Info dict ──
        info = {
            "lat": lat, "lon": lon, "alt": alt,
            "speed": v_air, "mach": mach, "q_dyn": q_dyn,
            "alpha_deg": alpha_deg,
            "thrust": thrust_mag, "drag": F_drag,
            "mdot_total": mdot_total,
            "mdot_s1": mdot_s1, "mdot_s2": mdot_s2, "mdot_srb": mdot_srb,
        }

        return f_total, tau_total, info

    def _get_Iyy(self):
        """Current pitch moment of inertia."""
        if self._s2_active or (not self._s1_active):
            return self._Iyy_s2 * (self._mass / 36000)
        return self._Iyy_s1 * (self._mass / self.propulsion.liftoff_mass)

    def _derivatives(self, t, state):
        """ODE right-hand side: d(state)/dt."""
        r = state[0:3]
        v = state[3:6]
        q = quat_normalize(state[6:10])
        w_body = state[10:13]

        # Forces
        f_total, tau_total, info = self._compute_forces(t, state)

        # Translational: F = ma in ECEF (non-rotating... we add Coriolis)
        a_inertial = f_total / self._mass

        # Coriolis and centrifugal (ECEF is rotating frame)
        omega_e = np.array([0.0, 0.0, OMEGA_EARTH])
        a_coriolis = -2 * np.cross(omega_e, v)
        a_centrifugal = -np.cross(omega_e, np.cross(omega_e, r))
        a_total = a_inertial + a_coriolis + a_centrifugal

        # Rotational kinematics: dq/dt = 0.5 * q ⊗ [0, wx, wy, wz]
        omega_quat = np.array([0.0, w_body[0], w_body[1], w_body[2]])
        dqdt = 0.5 * quat_multiply(q, omega_quat)

        # Rotational dynamics: I * dw/dt = tau - w × (I * w)
        Iyy = self._get_Iyy()
        I_body = np.array([Iyy * 0.1, Iyy, Iyy])  # Ixx << Iyy ≈ Izz for rocket
        Iw = I_body * w_body
        dwdt = (tau_total - np.cross(w_body, Iw)) / I_body

        # State derivative
        dsdt = np.zeros(13)
        dsdt[0:3] = v
        dsdt[3:6] = a_total
        dsdt[6:10] = dqdt
        dsdt[10:13] = dwdt

        return dsdt

    def _process_events(self, t, state):
        """Check and process flight events."""
        lat, lon, alt = ecef_to_geodetic(state[0:3])
        v_air = np.linalg.norm(state[3:6])
        state_dict = {"alt": alt, "speed": v_air, "mass": self._mass}

        for name, event in self.events.items():
            if event.check(t, state_dict):
                event.triggered = True
                event.t_actual = t
                self._handle_event(name, t, state)
                self.events_log.append({
                    "name": name, "t": t,
                    "alt": alt, "speed": v_air, "mass": self._mass
                })

    def _handle_event(self, name: str, t: float, state: np.ndarray):
        """Execute event actions (mass changes, stage flags)."""
        if name == "SRB_BURNOUT":
            self._srb_active = False

        elif name == "SRB_SEP":
            self._srb_separated = True
            srb_dry = sum(b.dry_mass for b in self.propulsion.srb_boosters)
            self._mass -= srb_dry

        elif name == "FAIRING_SEP":
            self._fairing_on = False
            self._mass -= self.propulsion.fairing_mass

        elif name == "MECO":
            self._s1_active = False

        elif name == "STAGE_SEP":
            # Set mass to S2 + payload
            self._mass = (self._s2_prop_remaining
                          + self.propulsion.s2_dry
                          + self.payload_mass)
            if self._fairing_on:
                self._mass += self.propulsion.fairing_mass

            # Reset attitude to align with velocity
            v = state[3:6]
            v_norm = np.linalg.norm(v)
            if v_norm > 1.0:
                v_hat = v / v_norm
                # Build rotation: body_x = v_hat
                up_approx = np.array([0, 0, 1.0])
                body_y = np.cross(v_hat, up_approx)
                bn = np.linalg.norm(body_y)
                if bn > 1e-6:
                    body_y /= bn
                else:
                    body_y = np.array([0, 1, 0])
                body_z = np.cross(v_hat, body_y)
                R_be = np.column_stack([v_hat, body_y, body_z])
                q_new = self._rotmat_to_quat(R_be)
                state[6:10] = q_new
            state[10:13] = 0.0  # Zero angular velocity

        elif name == "S2_IGNITION":
            self._s2_active = True

        elif name == "SECO":
            self._s2_active = False

    def _attitude_guidance(self, t, state):
        """
        Pitch schedule guidance: piecewise-linear interpolation.
        """
        q = quat_normalize(state[6:10])
        body_x, _, _ = self._get_body_axes(q)

        # Current pitch = angle of body_x from local horizontal
        r = state[0:3]
        lat, lon, alt = ecef_to_geodetic(r)
        R_enu = ecef_to_enu_matrix(lat, lon)

        body_x_enu = R_enu @ body_x
        pitch_current = np.degrees(np.arctan2(body_x_enu[2],
                                               np.sqrt(body_x_enu[0]**2 + body_x_enu[1]**2)))

        # Interpolate target pitch from schedule
        sched = self._pitch_schedule
        if t <= sched[0][0]:
            pitch_target = sched[0][1]
        elif t >= sched[-1][0]:
            pitch_target = sched[-1][1]
        else:
            for i in range(len(sched) - 1):
                if sched[i][0] <= t < sched[i+1][0]:
                    frac = (t - sched[i][0]) / (sched[i+1][0] - sched[i][0])
                    pitch_target = sched[i][1] + frac * (sched[i+1][1] - sched[i][1])
                    break

        return pitch_target, pitch_current

    def run(self, t_end: float = 1000.0, dt: float = 0.1) -> dict:
        """
        Run 6DOF simulation.

        Args:
            t_end: End time [s]
            dt: Integration timestep [s]

        Returns:
            dict with telemetry arrays and events
        """
        state = self._initial_state()
        t = 0.0
        last_record = -self.dt_record

        # Launch site ECEF for downrange calculation
        r_launch = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)

        # Thermal state
        T_nose = 300.0

        print(f"{self.config} 6DOF Simulation")
        print(f"  Liftoff mass: {self._mass/1000:.1f} ton")
        print(f"  Launch site: {self.lat0:.4f}°N, {self.lon0:.4f}°E")
        print(f"  Azimuth: {self.launch_azimuth:.0f}°")
        print("  Running...", flush=True)

        while t <= t_end:
            # Process events
            self._process_events(t, state)

            # Attitude guidance
            pitch_target, pitch_current = self._attitude_guidance(t, state)

            # Integration step (RK4)
            state = self._rk4_step(t, state, dt, pitch_target, pitch_current)
            t += dt

            # Update propellant
            _, _, info = self._compute_forces(t, state)
            self._s1_prop_remaining -= info["mdot_s1"] * dt
            self._s2_prop_remaining -= info["mdot_s2"] * dt
            self._srb_prop_remaining -= info["mdot_srb"] * dt
            self._s1_prop_remaining = max(self._s1_prop_remaining, 0)
            self._s2_prop_remaining = max(self._s2_prop_remaining, 0)
            self._srb_prop_remaining = max(self._srb_prop_remaining, 0)

            # Update mass
            self._mass -= info["mdot_total"] * dt
            self._mass = max(self._mass, self.propulsion.s2_dry + self.payload_mass)

            # Thermal update (simplified lumped capacitance)
            if info["q_dyn"] > 100 and info["alt"] < 120_000:
                rho = get_atmosphere(info["alt"])["rho"]
                # Sutton-Graves stagnation heating [W/m²]
                r_nose = 1.0  # nose radius [m]
                q_aero = 1.7415e-4 * np.sqrt(rho / r_nose) * info["speed"]**3
                q_aero = min(q_aero, 1e6)  # cap at 1 MW/m²
                q_rad = 5.67e-8 * 0.85 * T_nose**4
                rho_cp_d = 2500 * 1200 * 0.01  # ρ·cp·δ for TPS
                dT = (q_aero - q_rad) / rho_cp_d * dt
                T_nose += dT
                T_nose = np.clip(T_nose, 200, 2500)
            elif info["alt"] > 120_000:
                # Radiative cooling in vacuum
                q_rad = 5.67e-8 * 0.85 * T_nose**4
                rho_cp_d = 2500 * 1200 * 0.01
                T_nose -= q_rad / rho_cp_d * dt
                T_nose = max(T_nose, 200)

            # Normalize quaternion
            state[6:10] = quat_normalize(state[6:10])

            # Record telemetry
            if t - last_record >= self.dt_record:
                lat, lon, alt = ecef_to_geodetic(state[0:3])
                v_air = np.linalg.norm(state[3:6])
                # Inertial velocity = ECEF vel + Earth rotation
                omega_e = np.array([0.0, 0.0, OMEGA_EARTH])
                v_eci = state[3:6] + np.cross(omega_e, state[0:3])
                v_inertial_mag = np.linalg.norm(v_eci)
                atm = get_atmosphere(alt)
                mach = v_air / atm["a"] if atm["a"] > 1 else 0
                q_dyn = 0.5 * atm["rho"] * v_air**2
                accel = np.linalg.norm(info.get("thrust", 0) * np.ones(1)) / self._mass / G0 if self._mass > 0 else 0

                # Flight path angle in ENU
                R_enu = ecef_to_enu_matrix(lat, lon)
                v_enu = R_enu @ state[3:6]
                gamma = np.degrees(np.arctan2(v_enu[2],
                                               np.sqrt(v_enu[0]**2 + v_enu[1]**2)))

                roll, pitch, yaw = quat_to_euler(state[6:10])

                # Downrange distance
                dr = np.linalg.norm(state[0:3] - r_launch) / 1000  # [km]

                # Phase label
                if self._s2_active:
                    phase = "S2_BURN"
                elif self.events.get("STAGE_SEP", FlightEvent6DOF("", 0)).triggered:
                    phase = "S2_COAST"
                elif self._s1_active and self._srb_active:
                    phase = "S1+SRB"
                elif self._s1_active:
                    phase = "S1_ONLY"
                else:
                    phase = "COAST"

                self.telemetry.t.append(t)
                self.telemetry.lat.append(lat)
                self.telemetry.lon.append(lon)
                self.telemetry.alt.append(alt)
                self.telemetry.speed.append(v_air)
                self.telemetry.mach.append(mach)
                self.telemetry.q_dyn.append(q_dyn)
                self.telemetry.accel_g.append(info["thrust"] / (self._mass * G0) if self._mass > 0 else 0)
                self.telemetry.mass.append(self._mass)
                self.telemetry.thrust.append(info["thrust"])
                self.telemetry.drag.append(info["drag"])
                self.telemetry.gamma.append(gamma)
                self.telemetry.pitch.append(pitch)
                self.telemetry.roll.append(roll)
                self.telemetry.yaw.append(yaw)
                self.telemetry.alpha.append(info["alpha_deg"])
                self.telemetry.T_nose.append(T_nose)
                self.telemetry.downrange.append(dr)
                self.telemetry.v_inertial.append(v_inertial_mag)
                self.telemetry.phase.append(phase)

                last_record = t

            # Early termination: altitude below 0 after launch
            lat_c, lon_c, alt_c = ecef_to_geodetic(state[0:3])
            if t > 10 and alt_c < -1000:
                print(f"  [!] Impact at T+{t:.1f}s")
                break

            # SECO reached
            if self.events.get("SECO") and self.events["SECO"].triggered:
                # Coast for a bit then stop
                if t > self.events["SECO"].t_actual + 30:
                    break

        # Print summary
        self._print_summary()

        return {
            "telemetry": self.telemetry.to_dict(),
            "events": self.events_log,
            "config": self.config,
        }

    def _rk4_step(self, t, state, dt, pitch_target, pitch_current):
        """RK4 integration with attitude guidance."""
        # Apply pitch guidance as angular velocity command
        pitch_error = pitch_target - pitch_current
        pitch_rate_cmd = np.clip(pitch_error * 0.5, -3.0, 3.0)  # deg/s
        pitch_rate_rad = np.radians(pitch_rate_cmd)

        k1 = self._derivatives(t, state)
        k1[11] = pitch_rate_rad  # Override wy (pitch rate) with guidance

        s2 = state + 0.5 * dt * k1
        s2[6:10] = quat_normalize(s2[6:10])
        k2 = self._derivatives(t + 0.5*dt, s2)
        k2[11] = pitch_rate_rad

        s3 = state + 0.5 * dt * k2
        s3[6:10] = quat_normalize(s3[6:10])
        k3 = self._derivatives(t + 0.5*dt, s3)
        k3[11] = pitch_rate_rad

        s4 = state + dt * k3
        s4[6:10] = quat_normalize(s4[6:10])
        k4 = self._derivatives(t + dt, s4)
        k4[11] = pitch_rate_rad

        state_new = state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        state_new[6:10] = quat_normalize(state_new[6:10])
        return state_new

    def _print_summary(self):
        """Print mission summary."""
        telem = self.telemetry
        if not telem.t:
            return

        print("\n" + "=" * 60)
        print(f"{self.config} 6DOF Mission Summary")
        print("=" * 60)

        # Events
        print("\nFlight Events:")
        for ev in self.events_log:
            print(f"  T+{ev['t']:7.1f}s  {ev['name']:20s}  "
                  f"h={ev['alt']/1000:.1f}km  V={ev['speed']:.0f}m/s")

        # Key metrics
        idx_maxq = np.argmax(telem.q_dyn)
        idx_maxg = np.argmax(telem.accel_g)
        idx_maxt = np.argmax(telem.T_nose)

        print(f"\nMax-Q:       {max(telem.q_dyn)/1000:.1f} kPa at T+{telem.t[idx_maxq]:.0f}s")
        print(f"Max Accel:   {max(telem.accel_g):.1f} G at T+{telem.t[idx_maxg]:.0f}s")
        print(f"Max T_nose:  {max(telem.T_nose):.0f} K at T+{telem.t[idx_maxt]:.0f}s")
        print(f"Final V:     {telem.speed[-1]:.0f} m/s (inertial: {telem.v_inertial[-1]:.0f} m/s)")
        print(f"Final Alt:   {telem.alt[-1]/1000:.1f} km")
        print(f"Final Mass:  {telem.mass[-1]:.0f} kg")
        print(f"Ground Track: {telem.lat[0]:.2f}°N → {telem.lat[-1]:.2f}°N, "
              f"{telem.lon[0]:.2f}°E → {telem.lon[-1]:.2f}°E")
        print(f"Downrange:   {telem.downrange[-1]:.0f} km")


# ══════════════════════════════════════════════════════════════
# Monte Carlo Dispersion Analysis
# ══════════════════════════════════════════════════════════════

@dataclass
class MCDispersion:
    """Monte Carlo dispersion parameters (3-sigma values)."""
    thrust_3sigma: float = 0.02       # ±2% thrust variation
    isp_3sigma: float = 0.01          # ±1% Isp variation
    cd_3sigma: float = 0.05           # ±5% drag coefficient
    mass_3sigma: float = 0.005        # ±0.5% mass variation
    wind_speed_3sigma: float = 15.0   # ±15 m/s wind (at Max-Q altitude)
    azimuth_3sigma: float = 0.5       # ±0.5° launch azimuth


class H3MonteCarlo:
    """
    Monte Carlo dispersion analysis for H3 6DOF.

    Perturbs key parameters and runs N simulations to characterize
    trajectory dispersion (altitude, velocity, downrange at SECO).
    """

    def __init__(self, config: str = "H3-22S",
                 dispersion: MCDispersion = None,
                 seed: int = 42):
        self.config = config
        self.disp = dispersion or MCDispersion()
        self.rng = np.random.default_rng(seed)

    def run(self, n_runs: int = 50, dt: float = 0.1,
            t_end: float = 1000.0, verbose: bool = True) -> dict:
        """
        Run Monte Carlo analysis.

        Args:
            n_runs: Number of simulation runs
            dt: Integration timestep (larger for speed)
            t_end: Max simulation time
            verbose: Print progress

        Returns:
            dict with 'nominal', 'runs' (list of results), 'statistics'
        """
        if verbose:
            print(f"Monte Carlo: {n_runs} runs for {self.config}")
            print(f"  Dispersions: thrust={self.disp.thrust_3sigma*100:.0f}%, "
                  f"Isp={self.disp.isp_3sigma*100:.0f}%, "
                  f"Cd={self.disp.cd_3sigma*100:.0f}%")

        # Nominal run
        if verbose:
            print("  [0] Nominal...", flush=True)
        nom_sim = H3SixDOF(config=self.config)
        nom_result = nom_sim.run(t_end=t_end, dt=dt)

        # Dispersed runs
        run_summaries = []
        all_telemetry = []

        for i in range(n_runs):
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_runs}]...", flush=True)

            sim = H3SixDOF(config=self.config, dt_record=2.0)
            self._apply_dispersions(sim)

            # Suppress output
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                result = sim.run(t_end=t_end, dt=dt)
            finally:
                sys.stdout = old_stdout

            telem = result["telemetry"]
            events = result["events"]

            # Extract KPIs
            summary = self._extract_kpis(telem, events)
            run_summaries.append(summary)
            all_telemetry.append(telem)

        # Compute statistics
        stats = self._compute_stats(run_summaries)

        if verbose:
            self._print_stats(stats, n_runs)

        return {
            "nominal": nom_result,
            "runs": run_summaries,
            "all_telemetry": all_telemetry,
            "statistics": stats,
        }

    def _apply_dispersions(self, sim: H3SixDOF):
        """Apply random dispersions to simulation parameters."""
        d = self.disp
        rng = self.rng

        # Thrust dispersion
        thrust_factor = 1.0 + rng.normal(0, d.thrust_3sigma / 3)
        for e in sim.propulsion.le9_engines:
            e.thrust_vac *= thrust_factor
            e.thrust_sl *= thrust_factor
            e.__post_init__()
        for b in sim.propulsion.srb_boosters:
            b.thrust_avg *= thrust_factor
            b.__post_init__()
        if sim.propulsion.le5b_engine:
            sim.propulsion.le5b_engine.thrust_vac *= thrust_factor
            sim.propulsion.le5b_engine.__post_init__()

        # Isp dispersion
        isp_factor = 1.0 + rng.normal(0, d.isp_3sigma / 3)
        for e in sim.propulsion.le9_engines:
            e.isp_vac *= isp_factor
            e.isp_sl *= isp_factor
            e.__post_init__()
        for b in sim.propulsion.srb_boosters:
            b.isp_sl *= isp_factor
            b.isp_vac *= isp_factor
        if sim.propulsion.le5b_engine:
            sim.propulsion.le5b_engine.isp_vac *= isp_factor
            sim.propulsion.le5b_engine.__post_init__()

        # Drag dispersion
        cd_factor = 1.0 + rng.normal(0, d.cd_3sigma / 3)
        sim._cd_dispersion = cd_factor  # Applied in force computation

        # Mass dispersion
        mass_factor = 1.0 + rng.normal(0, d.mass_3sigma / 3)
        sim._mass *= mass_factor

        # Launch azimuth dispersion
        sim.launch_azimuth += rng.normal(0, d.azimuth_3sigma / 3)

    def _extract_kpis(self, telem: dict, events: list) -> dict:
        """Extract key performance indicators from a run."""
        kpi = {}
        if len(telem.get("t", [])) == 0:
            return kpi

        t = telem["t"]
        alt = telem["alt"]
        speed = telem["speed"]
        v_i = telem["v_inertial"]
        q_dyn = telem["q_dyn"]
        lat = telem["lat"]
        lon = telem["lon"]
        mass = telem["mass"]
        dr = telem["downrange"]

        kpi["max_q_kpa"] = np.max(q_dyn) / 1000
        kpi["t_max_q"] = t[np.argmax(q_dyn)]
        kpi["max_accel_g"] = np.max(telem["accel_g"])

        # SECO values
        seco_events = [e for e in events if e["name"] == "SECO"]
        if seco_events:
            t_seco = seco_events[0]["t"]
            idx = np.argmin(np.abs(t - t_seco))
            kpi["t_seco"] = t_seco
            kpi["alt_seco"] = alt[idx]
            kpi["v_seco"] = speed[idx]
            kpi["vi_seco"] = v_i[idx]
            kpi["lat_seco"] = lat[idx]
            kpi["lon_seco"] = lon[idx]
            kpi["mass_seco"] = mass[idx]
            kpi["dr_seco"] = dr[idx]
        else:
            # Use final values
            kpi["t_seco"] = t[-1]
            kpi["alt_seco"] = alt[-1]
            kpi["v_seco"] = speed[-1]
            kpi["vi_seco"] = v_i[-1]
            kpi["lat_seco"] = lat[-1]
            kpi["lon_seco"] = lon[-1]
            kpi["mass_seco"] = mass[-1]
            kpi["dr_seco"] = dr[-1]

        # MECO values
        meco_events = [e for e in events if e["name"] == "MECO"]
        if meco_events:
            t_meco = meco_events[0]["t"]
            idx = np.argmin(np.abs(t - t_meco))
            kpi["t_meco"] = t_meco
            kpi["alt_meco"] = alt[idx]
            kpi["v_meco"] = speed[idx]

        return kpi

    def _compute_stats(self, runs: list) -> dict:
        """Compute mean, std, min, max for each KPI."""
        if not runs:
            return {}

        keys = set()
        for r in runs:
            keys.update(r.keys())

        stats = {}
        for k in keys:
            values = [r[k] for r in runs if k in r]
            if values:
                arr = np.array(values)
                stats[k] = {
                    "mean": np.mean(arr),
                    "std": np.std(arr),
                    "min": np.min(arr),
                    "max": np.max(arr),
                    "p05": np.percentile(arr, 5),
                    "p95": np.percentile(arr, 95),
                }
        return stats

    def _print_stats(self, stats: dict, n_runs: int):
        """Print Monte Carlo statistics."""
        print(f"\n{'='*65}")
        print(f"Monte Carlo Results ({n_runs} runs)")
        print(f"{'='*65}")

        fmt = "  {:<20s}  {:>8s} ± {:>6s}  [{:>8s}, {:>8s}]"
        print(fmt.format("Parameter", "Mean", "1σ", "P5", "P95"))
        print("  " + "-" * 60)

        def pf(key, label, scale=1.0, unit=""):
            if key in stats:
                s = stats[key]
                print(f"  {label:<20s}  {s['mean']*scale:8.1f} ± {s['std']*scale:6.1f}"
                      f"  [{s['p05']*scale:8.1f}, {s['p95']*scale:8.1f}] {unit}")

        pf("max_q_kpa", "Max-Q", 1, "kPa")
        pf("max_accel_g", "Max Accel", 1, "G")
        pf("t_meco", "T_MECO", 1, "s")
        pf("v_meco", "V_MECO", 1e-3, "km/s")
        pf("alt_meco", "h_MECO", 1e-3, "km")
        pf("t_seco", "T_SECO", 1, "s")
        pf("v_seco", "V_SECO", 1e-3, "km/s")
        pf("vi_seco", "V_SECO (inertial)", 1e-3, "km/s")
        pf("alt_seco", "h_SECO", 1e-3, "km")
        pf("dr_seco", "Downrange", 1, "km")
        pf("mass_seco", "Mass_SECO", 1, "kg")
        pf("lat_seco", "Lat_SECO", 1, "°N")
        pf("lon_seco", "Lon_SECO", 1, "°E")


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "mc":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        mc = H3MonteCarlo(config="H3-22S")
        mc_result = mc.run(n_runs=n, dt=0.1)
        # Save for dashboard
        np.savez("results/mc_results.npz",
                 **{k: v for k, v in mc_result["statistics"].items()})
        print(f"\nSaved to results/mc_results.npz")
    else:
        sim = H3SixDOF(config="H3-22S", payload_mass=4000)
        result = sim.run(t_end=1000, dt=0.1)
        telem = result["telemetry"]
        np.savez("results/sixdof_telemetry.npz", **telem)
        print(f"\nTelemetry: {len(telem['t'])} samples")
        print(f"Saved to results/sixdof_telemetry.npz")
