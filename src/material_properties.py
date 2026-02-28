# Material Properties for H3 Fairing Sandwich Panel
# Based on typical aerospace composite structures (e.g., SLS fairing, H-IIA)

class Material:
    def __init__(self, name, E, nu, rho, type="isotropic"):
        self.name = name
        self.E = E      # Young's Modulus (Pa)
        self.nu = nu    # Poisson's Ratio
        self.rho = rho  # Density (kg/m^3)
        self.type = type

class OrthotropicMaterial(Material):
    def __init__(self, name, E1, E2, G12, nu12, rho):
        super().__init__(name, None, None, rho, type="orthotropic")
        self.E1 = E1    # Longitudinal Modulus
        self.E2 = E2    # Transverse Modulus
        self.G12 = G12  # Shear Modulus
        self.nu12 = nu12 # Major Poisson's Ratio
        
        # Derived properties
        self.nu21 = nu12 * E2 / E1

# ==============================================================================
# 1. CFRP Face Sheet (T700/Epoxy equivalent)
# ==============================================================================
# Typical values for aerospace grade CFRP (Unidirectional or Woven)
# Assuming Quasi-Isotropic Layup [0/45/-45/90] for macro behavior, 
# but here defining the lamina properties for detailed simulation.
CFRP_T700 = OrthotropicMaterial(
    name="CFRP_T700_Epoxy",
    E1=135e9,   # 135 GPa (Fiber direction)
    E2=10e9,    # 10 GPa (Transverse)
    G12=5e9,    # 5 GPa
    nu12=0.3,
    rho=1600.0  # 1600 kg/m^3
)

# ==============================================================================
# 2. Aluminum Honeycomb Core (Al 5052/5056)
# ==============================================================================
# Honeycomb is highly anisotropic.
# Out-of-plane (T direction) is stiff, In-plane (L, W) is compliant.
# Here we define effective continuum properties.
AL_HONEYCOMB = OrthotropicMaterial(
    name="Al_Honeycomb_Core",
    E1=1.0e6,    # In-plane modulus (very low) ~1 MPa
    E2=1.0e6,    # In-plane modulus (very low) ~1 MPa
    # Out-of-plane compression modulus is high, but for shell waves, shear matters
    G12=1.0e6,   # In-plane shear
    nu12=0.1,    # Almost zero Poisson effect in-plane
    rho=48.0     # 48 kg/m^3 (Core density, not bulk Al)
)
# Note: For transverse shear (G13, G23), values are much higher (~0.5 GPa)
AL_HONEYCOMB.G13 = 440e6 # 440 MPa
AL_HONEYCOMB.G23 = 220e6 # 220 MPa
AL_HONEYCOMB.E3 = 1.0e9  # Out-of-plane stiffness ~1 GPa

# ==============================================================================
# 3. Sandwich Panel Configuration
# ==============================================================================
class SandwichPanel:
    def __init__(self, face_material, core_material, face_thickness, core_thickness):
        self.face = face_material
        self.core = core_material
        self.t_f = face_thickness # meters
        self.t_c = core_thickness # meters
        
        # Calculate Effective Area Density (kg/m^2)
        self.area_density = 2 * self.face.rho * self.t_f + self.core.rho * self.t_c
        
        # Calculate Effective Bending Stiffness D (approximate)
        # D = E_f * t_f * (h^2) / 2  where h = t_c + t_f
        # Using E1 for simplicity
        h = self.t_c + self.t_f
        if hasattr(self.face, 'E1'):
            E_f = self.face.E1
        else:
            E_f = self.face.E
            
        self.D_bending = E_f * self.t_f * (h**2) / 2.0
        
        # Calculate Effective Shear Stiffness S
        # S = k * G_c * h
        # G_c is core shear modulus (G13 or G23)
        G_c = getattr(self.core, 'G13', 1e8) 
        self.S_shear = 1.0 * G_c * h # k=1.0 approx
        
    def get_wave_velocity(self, frequency):
        """
        Calculate flexural wave velocity (phase velocity) considering shear deformation.
        Using a simplified interpolation between Kirchhoff (Bending) and Shear mode.
        1/cp^2 = 1/cp_bend^2 + 1/cp_shear^2
        """
        omega = 2 * 3.14159 * frequency
        
        # 1. Kirchhoff Bending Velocity (Low Freq Limit)
        # cp_bend = (D / rho_A)^(1/4) * sqrt(omega)
        cp_bend = (self.D_bending / self.area_density)**0.25 * (omega)**0.5
        
        # 2. Shear Wave Velocity (High Freq Limit for Sandwich)
        # S = G_core * h
        # cp_shear = sqrt(S / rho_A)
        # This is the speed at which the core shears.
        cp_shear = (self.S_shear / self.area_density)**0.5
        
        # Combined (Series spring model for compliance -> velocities add like parallel resistors?)
        # Actually, for phase velocity dispersion of A0 mode in sandwich:
        # It transitions from bending behavior to shear behavior.
        # Heuristic: cp = 1 / sqrt(1/cp_bend^2 + 1/cp_shear^2)
        
        cp_combined = 1.0 / ( (1.0/cp_bend**2) + (1.0/cp_shear**2) )**0.5
        
        return cp_combined

# H3 Fairing Representative Panel
# Face: 1.0mm CFRP (approx 4-8 plies)
# Core: 30mm Al Honeycomb (typical for fairing)
# Updated G13 for Al Honeycomb to be realistic (e.g., Hexcel 3/8-5052-.003)
# G_L ~ 200-500 MPa, G_W ~ 100-250 MPa
AL_HONEYCOMB.G13 = 310e6 # 310 MPa (Typical 5052 core)

H3_FAIRING_PANEL = SandwichPanel(
    face_material=CFRP_T700,
    core_material=AL_HONEYCOMB,
    face_thickness=0.001,  # 1 mm
    core_thickness=0.030   # 30 mm
)

if __name__ == "__main__":
    print(f"--- H3 Fairing Sandwich Panel Properties ---")
    print(f"Face Sheet: {H3_FAIRING_PANEL.face.name} (t={H3_FAIRING_PANEL.t_f*1000}mm)")
    print(f"Core: {H3_FAIRING_PANEL.core.name} (t={H3_FAIRING_PANEL.t_c*1000}mm)")
    print(f"Area Density: {H3_FAIRING_PANEL.area_density:.2f} kg/m^2")
    print(f"Bending Stiffness D: {H3_FAIRING_PANEL.D_bending:.2e} Nm")
    
    freqs = [10e3, 50e3, 100e3, 300e3] # 10kHz to 300kHz (Lamb wave range)
    print("\n--- Dispersion Characteristics (Phase Velocity) ---")
    for f in freqs:
        cp = H3_FAIRING_PANEL.get_wave_velocity(f)
        print(f"Freq: {f/1000:.0f} kHz -> Phase Velocity: {cp:.1f} m/s")
