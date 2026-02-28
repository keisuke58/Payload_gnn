
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Output Directory
OUTPUT_DIR = "wiki_repo/images/defects"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# H3 Fairing Geometry (Type-S) - Reused
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

# Meshgrid for Nose (Tangent Ogive)
rho = (R**2 + H_NOSE**2) / (2*R)
xc = R - rho
T_nose, Z_nose = np.meshgrid(theta, z_nose)
R_nose = np.sqrt(rho**2 - (Z_nose - H_CYL)**2) + xc
X_nose = R_nose * np.cos(T_nose)
Y_nose = R_nose * np.sin(T_nose)

# Combine
X = np.vstack((X_cyl, X_nose))
Y = np.vstack((Y_cyl, Y_nose))
Z = np.vstack((Z_cyl, Z_nose))
T = np.vstack((T_cyl, T_nose))
Global_Z = Z

# ==============================================================================
# Defect Modeling (Debonding)
# ==============================================================================

def add_defect(field, T_grid, Z_grid, t_center, z_center, radius, amplitude):
    """
    Adds a Gaussian perturbation to simulate a debonding defect.
    """
    # Distance in (R*theta, z) space for simplicity
    # Approximate arc length distance
    R_local = R # Simplified, strictly R varies in nose
    dist_sq = (R_local * (T_grid - t_center))**2 + (Z_grid - z_center)**2
    
    perturbation = amplitude * np.exp(-dist_sq / (2 * radius**2))
    return field + perturbation

def add_fod(field, T_grid, Z_grid, t_center, z_center, radius, amplitude):
    """
    Adds a sharp peak to simulate Foreign Object Debris (FOD) / Hard Spot.
    Sharper decay than Gaussian (e.g., exponential).
    """
    R_local = R
    dist = np.sqrt((R_local * (T_grid - t_center))**2 + (Z_grid - z_center)**2)
    
    # Sharp exponential decay
    perturbation = amplitude * np.exp(-dist / (0.3 * radius))
    return field + perturbation

def add_impact(field, T_grid, Z_grid, t_center, z_center, radius, amplitude):
    """
    Adds a "Mexican Hat" (Ricker wavelet) shape to simulate Impact Damage.
    Central depression (negative) surrounded by pile-up (positive).
    """
    R_local = R
    dist_sq = (R_local * (T_grid - t_center))**2 + (Z_grid - z_center)**2
    sigma = radius / 2.0
    
    # Mexican Hat function
    # (1 - r^2/sigma^2) * exp(-r^2/2sigma^2)
    norm_dist_sq = dist_sq / (sigma**2)
    perturbation = amplitude * (1 - norm_dist_sq) * np.exp(-norm_dist_sq / 2)
    return field + perturbation

def plot_comparison(data_healthy, data_defective, title_prefix, filename_prefix, cmap='turbo'):
    """
    Plots Healthy, Defective, and Residual (Difference) side-by-side in 2D unfolded view.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Convert Theta to Arc Length at Base
    Arc_Length = T * R
    
    # Common extent
    extent = [Arc_Length.min(), Arc_Length.max(), Z.min(), Z.max()]
    
    # 1. Healthy
    im1 = axes[0].contourf(Arc_Length, Z, data_healthy, 100, cmap=cmap)
    axes[0].set_title(f"{title_prefix} (Healthy)", fontsize=14)
    axes[0].set_xlabel('Arc Length [mm]')
    axes[0].set_ylabel('Height Z [mm]')
    fig.colorbar(im1, ax=axes[0], shrink=0.6)
    
    # 2. Defective
    im2 = axes[1].contourf(Arc_Length, Z, data_defective, 100, cmap=cmap)
    axes[1].set_title(f"{title_prefix} (Defective)", fontsize=14)
    axes[1].set_xlabel('Arc Length [mm]')
    fig.colorbar(im2, ax=axes[1], shrink=0.6)
    
    # 3. Residual (Difference)
    residual = data_defective - data_healthy
    # Use diverging colormap for residual
    limit = np.max(np.abs(residual)) if np.max(np.abs(residual)) > 0 else 1.0
    im3 = axes[2].contourf(Arc_Length, Z, residual, 100, cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[2].set_title(f"Residual (Defect Signature)", fontsize=14)
    axes[2].set_xlabel('Arc Length [mm]')
    fig.colorbar(im3, ax=axes[2], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {filename_prefix}_comparison.png")

def plot_3d_defect(data, title, filename, cmap='turbo'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.get_cmap(cmap)(data), 
                           rstride=2, cstride=2, shade=True, alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_box_aspect((1, 1, 3))
    
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(data)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# ------------------------------------------------------------------------------
# Base Fields (Healthy)
# ------------------------------------------------------------------------------
# 1. Displacement (Max Q)
disp_base = 3.0 * np.sin(np.pi * Z / H_TOTAL * 0.8) * (1 + 0.2*np.cos(3*T))
disp_base += 1.5 * (Z / H_TOTAL)**2 * np.cos(T)

# 2. Stress (Von Mises)
stress_base = 75.0 * ((H_TOTAL - Z) / H_TOTAL)**1.5
stress_base += 5.0 * np.cos(3 * T) * (Z / H_TOTAL)

# ------------------------------------------------------------------------------
# Introduce Defects (Scenario: Multi-Defect Fairing)
# ------------------------------------------------------------------------------
# Base fields
disp_multi = disp_base.copy()
stress_multi = stress_base.copy()

# 1. Debonding (Large) - Cylinder
# Location: Theta ~ pi/6 (30 deg), Z ~ 2500mm
# Effect: Smooth bulge
disp_multi = add_defect(disp_multi, T, Z, np.pi/6, 2500, 300, 0.8) 

# 2. Foreign Object Debris (FOD) - Upper Cylinder
# Location: Theta ~ pi/3 (60 deg), Z ~ 4000mm
# Effect: Sharp localized stress concentration (hard point)
stress_multi = add_fod(stress_multi, T, Z, np.pi/3, 4000, 100, 25.0) # +25 MPa sharp peak

# 3. Impact Damage - Nose Cone
# Location: Theta ~ pi/12 (15 deg), Z ~ 7000mm
# Effect: Dent (Mexican Hat shape in displacement)
disp_multi = add_impact(disp_multi, T, Z, np.pi/12, 7000, 150, -0.5) # -0.5mm dent (plus pileup)


# ------------------------------------------------------------------------------
# Generate Plots
# ------------------------------------------------------------------------------

# 1. Displacement Comparison (Debonding + Impact)
plot_comparison(disp_base, disp_multi, "Displacement (Debonding & Impact)", "displacement_multi")

# 2. Stress Comparison (Debonding + FOD)
plot_comparison(stress_base, stress_multi, "Stress (Debonding & FOD)", "stress_multi")

# 3. 3D Visualization of Combined Defect Signature
residual_disp = disp_multi - disp_base
# Normalize residual for 3D color mapping (0-1)
res_norm = (residual_disp - residual_disp.min()) / (residual_disp.max() - residual_disp.min())
plot_3d_defect(res_norm, "Combined Defect Signatures (3D)", "defect_3d_multi.png", cmap='plasma')

print("Defect visualizations generated successfully.")
