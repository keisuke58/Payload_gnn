import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================================
# PINN (Physics-Informed Neural Network) for Inverse Wave Problem
# Goal: Find wave speed c(x) given sparse measurements of u(x,t)
# Wave Equation: u_tt - (c(x)^2 * u_x)_x = 0
# Defect: Region where c(x) != c0
# ==============================================================================

class PINN(nn.Module):
    def __init__(self, layers_u, layers_c):
        super(PINN, self).__init__()
        
        # Network for u(x,t)
        self.u_net = self._build_net(layers_u)
        
        # Network for c(x) (Wave speed distribution)
        self.c_net = self._build_net(layers_c)
        
    def _build_net(self, layers):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh()) # Tanh is good for higher-order derivatives
        return nn.Sequential(*modules)
        
    def forward(self, x, t):
        # Predict u(x,t)
        inputs = torch.cat([x, t], dim=1)
        u = self.u_net(inputs)
        
        # Predict c(x) - independent of t
        c = self.c_net(x)
        return u, c

def physics_loss(model, x, t):
    """
    Compute residual of the wave equation:
    Residual = u_tt - c^2 * u_xx - 2*c*c_x*u_x
    Actually simpler form: u_tt - (c^2 * u_x)_x = 0
    """
    x.requires_grad = True
    t.requires_grad = True
    
    u, c = model(x, t)
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    
    # c(x) derivatives (if c varies spatially)
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    
    # Wave Equation: u_tt = c^2 * u_xx + 2*c*c_x*u_x
    # Or simply: u_tt = (c*u_x)_x * c ? No, standard is u_tt = c^2 * u_xx for constant c locally
    # For variable c(x): u_tt = d/dx ( c^2(x) * du/dx ) = c^2 * u_xx + 2*c*c_x * u_x
    
    residual = u_tt - (c**2 * u_xx + 2*c*c_x * u_x)
    return torch.mean(residual**2)

def generate_synthetic_data():
    """
    Simulate a 1D wave with a defect (slow speed region)
    Domain: x in [-1, 1], t in [0, 1]
    c(x) = 1.0 everywhere, except c(x)=0.5 in [0.2, 0.4] (Defect)
    """
    print("Generating synthetic wave data (Forward Finite Difference)...")
    
    nx, nt = 200, 200
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # True c(x)
    c_true = np.ones(nx)
    c_true[(x > 0.2) & (x < 0.4)] = 0.5 # Defect
    
    # Stability condition: c * dt / dx <= 1
    # 1.0 * (1/200) / (2/200) = 0.5 <= 1 (OK)
    
    u = np.zeros((nt, nx))
    
    # Initial Condition: Gaussian Pulse
    u[0, :] = np.exp(-100 * (x + 0.5)**2)
    u[1, :] = u[0, :] # Zero initial velocity
    
    # FDM Loop
    for n in range(1, nt-1):
        for i in range(1, nx-1):
            c2 = c_true[i]**2
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + (dt/dx)**2 * c2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
    # Sample sparse data (simulating sensors)
    # Sensors at x = -0.8, -0.4, 0.0, 0.4, 0.8
    sensor_indices = [20, 60, 100, 140, 180]
    
    X_sensors = []
    T_sensors = []
    U_sensors = []
    
    for idx in sensor_indices:
        for n in range(0, nt, 5): # Every 5th time step
            X_sensors.append(x[idx])
            T_sensors.append(t[n])
            U_sensors.append(u[n, idx])
            
    # Collocation points (randomly sampled in domain)
    N_col = 2000
    X_col = np.random.uniform(-1, 1, N_col)
    T_col = np.random.uniform(0, 1, N_col)
    
    return (
        np.array(X_sensors).reshape(-1, 1), 
        np.array(T_sensors).reshape(-1, 1), 
        np.array(U_sensors).reshape(-1, 1),
        X_col.reshape(-1, 1),
        T_col.reshape(-1, 1),
        x, c_true
    )

def train_pinn_prototype():
    # Data
    x_s, t_s, u_s, x_col, t_col, x_grid, c_true = generate_synthetic_data()
    
    # Convert to Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    x_s = torch.tensor(x_s, dtype=torch.float32).to(device)
    t_s = torch.tensor(t_s, dtype=torch.float32).to(device)
    u_s = torch.tensor(u_s, dtype=torch.float32).to(device)
    
    x_col = torch.tensor(x_col, dtype=torch.float32).to(device)
    t_col = torch.tensor(t_col, dtype=torch.float32).to(device)
    
    # Model
    # u_net: [x,t] -> [u] (2 -> 40 -> 40 -> 40 -> 1)
    # c_net: [x] -> [c]   (1 -> 20 -> 20 -> 1)
    model = PINN([2, 40, 40, 40, 1], [1, 20, 20, 1]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting PINN Training (Inverse Problem)...")
    start_time = time.time()
    
    EPOCHS = 500 # Reduced for prototype speed
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # 1. Data Loss (MSE at sensors)
        u_pred, _ = model(x_s, t_s)
        loss_data = torch.mean((u_pred - u_s)**2)
        
        # 2. Physics Loss (PDE residual at collocation points)
        loss_phy = physics_loss(model, x_col, t_col)
        
        # Total Loss
        loss = loss_data + 1.0 * loss_phy # Weighting might need tuning
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.6f} (Data: {loss_data.item():.6f}, Phy: {loss_phy.item():.6f})")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Evaluation: Predict c(x)
    x_test = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        _, c_pred = model(x_test, torch.zeros_like(x_test)) # t doesn't matter for c_net
        
    c_pred = c_pred.cpu().numpy().flatten()
    
    # Calculate Error in Defect Region
    defect_mask = (x_grid > 0.2) & (x_grid < 0.4)
    c_defect_true = 0.5
    c_defect_pred = np.mean(c_pred[defect_mask])
    
    error = np.abs(c_defect_pred - c_defect_true)
    print(f"\nDefect Region (0.2 < x < 0.4):")
    print(f"True Wave Speed: {c_defect_true}")
    print(f"Pred Wave Speed: {c_defect_pred:.4f}")
    print(f"Error: {error:.4f}")
    
    if error < 0.2:
        print("[SUCCESS] PINN identified the low-velocity defect region!")
    else:
        print("[WARNING] PINN struggled to identify the defect accurately. More training/points needed.")

if __name__ == "__main__":
    train_pinn_prototype()
