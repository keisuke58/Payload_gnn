
import math
import numpy as np

# H3 Type-S Fairing Constants
RADIUS = 2600.0       # mm
H_BARREL = 5000.0     # mm
H_NOSE = 5400.0       # mm
TOTAL_HEIGHT = 10400.0 # mm

# Material Properties (CFRP T1000G / Al-5052 Honeycomb)
# Face Sheet (CFRP)
FACE_T = 1.0          # mm
E_FACE = 160000.0     # MPa (E1, approx)
RHO_FACE = 1.60e-9    # tonne/mm^3 (1600 kg/m^3)
NU_FACE = 0.3

# Core (Honeycomb)
CORE_T = 38.0         # mm (Corrected from 30mm)
E_CORE = 1.0          # MPa (In-plane, negligible)
G_CORE = 400.0        # MPa (Shear modulus, approx)
RHO_CORE = 0.05e-9    # tonne/mm^3 (50 kg/m^3)

# 1. Geometry Calculation (Surface Area)
def calculate_surface_area():
    # Barrel
    area_barrel = 2 * math.pi * RADIUS * H_BARREL
    
    # Tangent Ogive (Approximate)
    # Using integration for surface of revolution
    # r(z) = sqrt(rho^2 - (z - H_BARREL)^2) + (R - rho)
    # This is complex, approximating as cone for rough estimate or using numeric integration
    # For H3 Type-S, Ogive surface area is approx 1.15 * Cone area
    # Cone area = pi * R * sqrt(R^2 + H_NOSE^2)
    area_cone = math.pi * RADIUS * math.sqrt(RADIUS**2 + H_NOSE**2)
    area_ogive = area_cone * 1.05 # Correction factor for convexity
    
    return area_barrel + area_ogive

# 2. Mass Calculation
def calculate_mass(area_mm2):
    vol_face = area_mm2 * FACE_T * 2 # Inner + Outer
    vol_core = area_mm2 * CORE_T
    
    mass_face = vol_face * RHO_FACE
    mass_core = vol_core * RHO_CORE
    
    return mass_face + mass_core

# 3. Stiffness Calculation (Sandwich Theory)
def calculate_stiffness():
    # D = E_f * t_f * d^2 / 2 (Simplified)
    # d = h_core + t_face
    d = CORE_T + FACE_T
    D = (E_FACE * FACE_T * d**2) / 2.0 / (1 - NU_FACE**2)
    return D

# 4. Buckling Load (Axial Compression for Cylinder)
def calculate_buckling_load(D):
    # P_cr = 2 * pi / sqrt(3(1-v^2)) * E * t^2  (Monocoque)
    # For Sandwich: P_cr_face_wrinkling, P_cr_global
    # Global Buckling (NASA SP-8007)
    # N_cr = (2/R) * sqrt(D * A_shear) ... simplified
    # Using simple equivalent cylinder approach:
    # EI_eq = D
    # P_cr is proportional to D
    
    # Critical Stress Resultant Nx (N/mm)
    # Nx_cr = (1/R) * sqrt(2 * D * E_FACE * 2 * FACE_T) # Very rough approx
    
    # Better: Classical Sandwich Shell Buckling
    # N_cr = (2 * E_face * t_face / sqrt(3*(1-v^2))) * (d / R)
    d = CORE_T + FACE_T
    N_cr = (2 * E_FACE * FACE_T / math.sqrt(3 * (1 - NU_FACE**2))) * (d / RADIUS)
    return N_cr

def main():
    area = calculate_surface_area()
    mass = calculate_mass(area) # Tonne
    mass_kg = mass * 1000.0
    
    D = calculate_stiffness()
    N_cr = calculate_buckling_load(D)
    
    print("--- H3 Fairing (Type-S) Theoretical Analysis ---")
    print(f"Geometry: R={RADIUS}mm, H_Cyl={H_BARREL}mm, H_Nose={H_NOSE}mm")
    print(f"Thickness: Face={FACE_T}mm x2, Core={CORE_T}mm")
    print(f"Total Surface Area: {area/1e6:.2f} m^2")
    print("-" * 30)
    print(f"Estimated Mass (CFRP+Core): {mass_kg:.2f} kg")
    print(f"  - Face Sheets: {area * FACE_T * 2 * RHO_FACE * 1000:.2f} kg")
    print(f"  - Core: {area * CORE_T * RHO_CORE * 1000:.2f} kg")
    print("-" * 30)
    print(f"Bending Stiffness (D): {D:.2e} N-mm")
    print(f"Critical Axial Load (N_cr): {N_cr:.2f} N/mm")
    print(f"  - Equivalent Axial Force: {N_cr * 2 * math.pi * RADIUS / 1000:.2f} kN")
    print("-" * 30)
    print("Note: These are theoretical estimates for the healthy baseline.")

if __name__ == "__main__":
    main()
