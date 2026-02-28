import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_point_cloud(pos, value, title="3D Point Cloud"):
    """
    Visualizes a 3D point cloud with color-coded values.
    pos: (N, 3) array
    value: (N,) array
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    img = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=value, cmap='viridis', s=5)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(img, ax=ax, label='Value (e.g. Wave Amplitude)')
    
    plt.show()

def plot_2d_heatmap(data, title="2D Heatmap"):
    """
    Visualizes a 2D grid data.
    data: (H, W) array
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='inferno', origin='lower')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_pinn_results(x, c_true, c_pred, title="PINN Inverse Analysis"):
    """
    Visualizes PINN prediction vs True Wave Speed.
    x: (N,) array
    c_true: (N,) array
    c_pred: (N,) array
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, c_true, 'k--', linewidth=2, label='True c(x)')
    plt.plot(x, c_pred, 'r-', linewidth=2, label='Predicted c(x)')
    
    # Highlight defect region
    plt.axvspan(0.2, 0.4, color='yellow', alpha=0.3, label='Defect Region')
    
    plt.title(title)
    plt.xlabel('Position x')
    plt.ylabel('Wave Speed c(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example Usage
    print("Generating example plots...")
    
    # 1. 3D Example
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.linspace(0, 10, 1000)
    x = 2.6 * np.cos(theta)
    y = 2.6 * np.sin(theta)
    pos = np.stack([x, y, z], axis=1)
    val = np.sin(z)
    try:
        plot_3d_point_cloud(pos, val, "Example 3D Cylinder")
        print("3D Plot generated.")
    except Exception as e:
        print(f"Could not generate 3D plot (no display?): {e}")

    # 2. 2D Example
    grid = np.random.rand(64, 64)
    try:
        plot_2d_heatmap(grid, "Example 2D FNO Input")
        print("2D Plot generated.")
    except Exception as e:
        print(f"Could not generate 2D plot: {e}")

    # 3. PINN Example
    x_line = np.linspace(0, 1, 100)
    c_t = np.ones(100)
    c_t[20:40] = 0.5
    c_p = np.ones(100) * 0.9
    c_p[20:40] = 0.6
    try:
        plot_pinn_results(x_line, c_t, c_p, "Example PINN Result")
        print("PINN Plot generated.")
    except Exception as e:
        print(f"Could not generate PINN plot: {e}")
