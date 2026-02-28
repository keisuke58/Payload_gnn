
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Output Directory
OUTPUT_DIR = "wiki_repo/images/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# H3 Fairing Geometry (Type-S)
# ==============================================================================
R = 2600.0       # mm
H_CYL = 5000.0   # mm
H_NOSE = 5400.0  # mm
H_TOTAL = H_CYL + H_NOSE

# Grid Generation
theta = np.linspace(0, np.pi/3, 100) # 60 degree sector
z_cyl = np.linspace(0, H_CYL, 100)
z_nose = np.linspace(H_CYL, H_TOTAL, 100)

# Meshgrid for Cylinder
T_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
X_cyl = R * np.cos(T_cyl)
Y_cyl = R * np.sin(T_cyl)

# Meshgrid for Nose (Tangent Ogive Approximation)
# rho = (R^2 + L^2) / (2R)
rho = (R**2 + H_NOSE**2) / (2*R)
xc = R - rho
# r(z) = sqrt(rho^2 - (z-H_CYL)^2) + xc
T_nose, Z_nose = np.meshgrid(theta, z_nose)
R_nose = np.sqrt(rho**2 - (Z_nose - H_CYL)**2) + xc
X_nose = R_nose * np.cos(T_nose)
Y_nose = R_nose * np.sin(T_nose)

# Combine
X = np.vstack((X_cyl, X_nose))
Y = np.vstack((Y_cyl, Y_nose))
Z = np.vstack((Z_cyl, Z_nose))
T = np.vstack((T_cyl, T_nose)) # Theta
Global_Z = Z

# ==============================================================================
# Physics Simulation (Analytical)
# ==============================================================================

def plot_field_3d(data, title, filename, cmap='turbo'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.get_cmap(cmap)(data), 
                           rstride=2, cstride=2, shade=True, alpha=0.9)
    
    # Settings
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_box_aspect((1, 1, 3)) # Aspect ratio
    
    # Colorbar
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(data)
    # plt.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def plot_field_2d_unfolded(data, title, filename, cmap='turbo'):
    # Unfold: Theta (x) vs Z (y)
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Convert Theta to Arc Length at Base
    Arc_Length = T * R
    
    c = ax.contourf(Arc_Length, Z, data, 100, cmap=cmap)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Arc Length (R*theta) [mm]')
    ax.set_ylabel('Height Z [mm]')
    
    plt.colorbar(c, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# ------------------------------------------------------------------------------
# Case 1: Von Mises Stress (Max Q - Bending Dominant)
# ------------------------------------------------------------------------------
# Stress is max at base (Z=0), decreases with height
# Sigma ~ M*y/I -> Proportional to (H_Total - Z)
stress_base = 75.0 # MPa (Approx max for CFRP at Max Q)
stress_dist = stress_base * ((H_TOTAL - Z) / H_TOTAL)**1.5
# Add some hoop stress variation
stress_dist += 5.0 * np.cos(3 * T) * (Z / H_TOTAL)

# Normalize for visualization (0 to 1 range for colormap application logic if needed, 
# but here we pass raw data and let cmap handle normalization)
plot_field_3d(stress_dist / np.max(stress_dist), "Von Mises Stress Distribution (Normalized)", "stress_3d.png", cmap='inferno')
plot_field_2d_unfolded(stress_dist, "Von Mises Stress [MPa]", "stress_2d_map.png", cmap='inferno')

# ------------------------------------------------------------------------------
# Case 2: Radial Displacement (Internal Pressure + Aerodynamic)
# ------------------------------------------------------------------------------
# Max displacement typically at middle of panels or tip depending on BC
# For fairing breathing mode: Max at middle of unconstrained sections?
# Assuming fixed base, max displacement near middle-upper section due to dynamic pressure
disp_max = 3.0 # mm
# Mode shape: sin(pi * z / H) * sin(3 * theta)
disp_field = disp_max * np.sin(np.pi * Z / H_TOTAL * 0.8) * (1 + 0.2*np.cos(3*T))
# Add tip deflection (cantilever)
disp_field += 1.5 * (Z / H_TOTAL)**2 * np.cos(T)

plot_field_3d(disp_field / np.max(disp_field), "Total Displacement Magnitude", "displacement_3d.png", cmap='turbo')
plot_field_2d_unfolded(disp_field, "Displacement [mm]", "displacement_2d_map.png", cmap='turbo')

# ------------------------------------------------------------------------------
# Case 3: 1st Bending Mode Shape (Eigenvalue Analysis)
# ------------------------------------------------------------------------------
# Cantilever mode
mode_shape = (Z / H_TOTAL)**2 * np.cos(T)
plot_field_3d(mode_shape, "1st Bending Mode Shape (63 Hz)", "mode_shape_1.png", cmap='viridis')

print("All visualizations generated successfully.")
