"""H3 Virtual Twin — Integrated simulation modules."""

from .propulsion import LE9Engine, LE5B3Engine, SRB3Booster, H3PropulsionSystem
from .aerodynamics import H3Aerodynamics, H3Geometry, atmosphere_isa
from .aerothermal import H3Aerothermal, FairingGeometry, TPSMaterial
from .attitude_control import H3AttitudeController, PIDController, GravityTurnProfile
from .orchestrator import H3FlightOrchestrator, FlightState, FlightEvent

__all__ = [
    # Propulsion (A1-A4)
    "LE9Engine", "LE5B3Engine", "SRB3Booster", "H3PropulsionSystem",
    # Aerodynamics (A2)
    "H3Aerodynamics", "H3Geometry", "atmosphere_isa",
    # Aerothermal (A5)
    "H3Aerothermal", "FairingGeometry", "TPSMaterial",
    # Attitude Control (A6)
    "H3AttitudeController", "PIDController", "GravityTurnProfile",
    # Orchestrator (B1-B4)
    "H3FlightOrchestrator", "FlightState", "FlightEvent",
]
