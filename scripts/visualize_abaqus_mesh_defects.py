import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Configuration
NODES_CSV = 'dataset_output_25mm_400/sample_0100/nodes.csv'
ELEMS_CSV = 'dataset_output_25mm_400/sample_0100/elements.csv'
OUTPUT_DIR = 'wiki_repo/images/defects_mesh'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading mesh from {NODES_CSV}...")

# 1. Load Nodes
try:
    nodes_df = pd.read_csv(NODES_CSV)
    # Columns: node_id,x,y,z,ux,uy,uz,temp,s11,s22,s12,smises,defect_label
except FileNotFoundError:
    print(f"Error: {NODES_CSV} not found. Please ensure dataset exists.")
    exit(1)

# Map node_id to index (0-based)
node_map = {nid: i for i, nid in enumerate(nodes_df['node_id'])}
coords = nodes_df[['x', 'y', 'z']].values
n_nodes = len(coords)
print(f"Loaded {n_nodes} nodes.")

# 2. Load Elements and Build Triangles
print(f"Loading elements from {ELEMS_CSV}...")
elems_df = pd.read_csv(ELEMS_CSV)
# Columns: elem_id,elem_type,n1,n2,n3,n4,mises_avg

triangles = []
for _, row in elems_df.iterrows():
    try:
        n1 = node_map.get(row['n1'])
        n2 = node_map.get(row['n2'])
        n3 = node_map.get(row['n3'])
        n4_raw = row['n4']
        
        if n1 is None or n2 is None or n3 is None:
            continue
            
        # Triangle 1 (1-2-3)
        triangles.append([n1, n2, n3])
        
        # Triangle 2 (1-3-4) for Quads
        if n4_raw != -1 and n4_raw in node_map:
            n4 = node_map[n4_raw]
            triangles.append([n1, n3, n4])
            
    except Exception as e:
        continue

triangles = np.array(triangles)
print(f"Constructed {len(triangles)} triangles for visualization.")

# 3. Calculate Synthetic Defect Fields
# Geometry: Y=axial, XZ=radial
X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]
R = np.sqrt(X**2 + Z**2)
Theta = np.arctan2(Z, X) # -pi to pi

# Normalize fields
disp_field = np.zeros(n_nodes)
stress_field = np.zeros(n_nodes)

def add_gaussian(field, theta_c, y_c, r_d, amp):
    """Gaussian perturbation (Debonding)"""
    # Arc distance approximation
    d_theta = np.abs(Theta - theta_c)
    d_theta = np.where(d_theta > np.pi, 2*np.pi - d_theta, d_theta)
    arc = R * d_theta
    dy = Y - y_c
    dist = np.sqrt(arc**2 + dy**2)
    return field + amp * np.exp(-dist**2 / (2 * (r_d/2)**2))

def add_mexican_hat(field, theta_c, y_c, r_d, amp):
    """Mexican Hat wavelet (Impact)"""
    d_theta = np.abs(Theta - theta_c)
    d_theta = np.where(d_theta > np.pi, 2*np.pi - d_theta, d_theta)
    arc = R * d_theta
    dy = Y - y_c
    dist_sq = arc**2 + dy**2
    sigma = r_d / 2.0
    norm_dist_sq = dist_sq / (sigma**2)
    return field + amp * (1 - norm_dist_sq) * np.exp(-norm_dist_sq / 2)

def add_sharp_peak(field, theta_c, y_c, r_d, amp):
    """Exponential decay (FOD)"""
    d_theta = np.abs(Theta - theta_c)
    d_theta = np.where(d_theta > np.pi, 2*np.pi - d_theta, d_theta)
    arc = R * d_theta
    dy = Y - y_c
    dist = np.sqrt(arc**2 + dy**2)
    return field + amp * np.exp(-dist / (0.3 * r_d))

# --- Defect Scenarios ---

# 1. Debonding (Bulge)
disp_field = add_gaussian(disp_field, np.pi/4, 2500, 400, 1.0)

# 2. Impact (Dent)
disp_field = add_mexican_hat(disp_field, -np.pi/4, 4000, 300, -0.8)

# 3. FOD (Stress Peak)
stress_field = add_sharp_peak(stress_field, np.pi, 3000, 150, 50.0)

# 4. Stress Concentration at Debonding Edge
# Derivative-like effect: Ring of stress
# Simulating via difference of Gaussians or just adding a ring
# Simple ring:
stress_field = add_gaussian(stress_field, np.pi/4, 2500, 400, 20.0) # Base stress
# Subtract slightly narrower to make a ring? No, just add peak for now.

# Visualization
print("Generating plots...")

def plot_2d_unrolled(triangles, theta, y, field, title, filename, cmap='viridis'):
    """2D Unrolled view (Theta vs Axial)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter triangles that cross the seam (pi/-pi)
    # If any edge length in theta > pi, it wraps around
    # We can just ignore them or split them. Ignoring is easier for viz.
    t1 = theta[triangles[:, 0]]
    t2 = theta[triangles[:, 1]]
    t3 = theta[triangles[:, 2]]
    
    mask = (np.abs(t1 - t2) < np.pi) & \
           (np.abs(t2 - t3) < np.pi) & \
           (np.abs(t3 - t1) < np.pi)
    
    valid_triangles = triangles[mask]
    
    trip = ax.tripcolor(theta, y, valid_triangles, field, cmap=cmap, shading='gouraud')
    fig.colorbar(trip, ax=ax, label='Magnitude')
    
    ax.set_title(title)
    ax.set_xlabel('Theta (rad)')
    ax.set_ylabel('Axial Position Y (mm)')
    ax.set_xlim(-np.pi, np.pi)
    
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=100)
    print(f"Saved {path}")
    plt.close()

def plot_3d_scatter(x, y, z, field, title, filename, cmap='viridis', stride=10):
    """3D Scatter plot (Subsampled)"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample
    xs = x[::stride]
    ys = y[::stride]
    zs = z[::stride]
    fs = field[::stride]
    
    sc = ax.scatter(xs, zs, ys, c=fs, cmap=cmap, s=1, alpha=0.5) # Y is up
    
    fig.colorbar(sc, ax=ax, shrink=0.5, label='Magnitude')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # Set view
    ax.view_init(elev=20, azim=45)
    
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=100)
    print(f"Saved {path}")
    plt.close()

# Plot Displacement (2D Unrolled)
plot_2d_unrolled(triangles, Theta, Y, disp_field, 
               'Displacement Magnitude (Unrolled) on Abaqus Mesh', 
               'abaqus_mesh_displacement_2d.png', cmap='jet')

# Plot Stress (2D Unrolled)
plot_2d_unrolled(triangles, Theta, Y, stress_field, 
               'Von Mises Stress (Unrolled) on Abaqus Mesh', 
               'abaqus_mesh_stress_2d.png', cmap='inferno')

# Plot Displacement (3D Scatter)
plot_3d_scatter(X, Y, Z, disp_field, 
               'Displacement (3D Point Cloud)', 
               'abaqus_mesh_displacement_3d.png', cmap='jet', stride=5)

print("Done.")
