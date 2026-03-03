import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from scipy.optimize import minimize

# Add src to path to import FNO model if available
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '../../src')
sys.path.append(src_path)

# Try to import FNO2d, else use mock
try:
    from models_fno import FNO2d
    print("Successfully imported FNO2d from src.models_fno")
except ImportError as e:
    print(f"Warning: Could not import FNO2d ({e}). Using Mock model.")
    class FNO2d(torch.nn.Module):
        def __init__(self, modes1, modes2, width, in_channels=3):
            super().__init__()
            self.width = width
        def forward(self, x):
            batch, c, sx, sy = x.shape
            return torch.randn(batch, 1, sx, sy)

def run_fno_inference(grid_size=64):
    """
    Run FNO inference (or mock) to generate a defect probability map.
    """
    print(f"Running FNO inference simulation on {grid_size}x{grid_size} grid...")
    
    # Initialize model
    model = FNO2d(modes1=12, modes2=12, width=32, in_channels=3)
    
    # Create dummy input: (Batch, Channels, X, Y)
    x = torch.randn(1, 3, grid_size, grid_size)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    # Create artificial defect
    center = grid_size // 2
    # Add a defect (16x16) to make p significant (~6%)
    out[0, 0, center-8:center+8, center-8:center+8] = 5.0
    
    # Convert to probability map using Sigmoid
    prob_map = torch.sigmoid(out).squeeze().numpy()
    
    return prob_map

def run_mlae_estimation(p_true, m_schedules=[0, 1, 2, 4], shots=1000):
    """
    Run Maximum Likelihood Amplitude Estimation (MLAE) to estimate p_true.
    Using Grover Operator Q, we amplify the amplitude of state |1>.
    """
    print(f"\n[Quantum MLAE] Target p={p_true:.6f}, Schedules={m_schedules}, Shots={shots}")
    
    # Base theta for p_true: p = sin^2(theta) => theta = arcsin(sqrt(p))
    theta_true = np.arcsin(np.sqrt(p_true))
    
    results = []
    
    backend = Aer.get_backend('aer_simulator')
    
    for m in m_schedules:
        # Create Circuit for Q^m A |0>
        # A = Ry(2*theta) -> State is cos(theta)|0> + sin(theta)|1>
        # Grover iteration rotates by 2*theta.
        # After m iterations, angle is (2m+1)*theta.
        # So probability of |1> is sin^2((2m+1)*theta).
        
        qc = QuantumCircuit(1, 1)
        # Apply (2m+1) rotations of 2*theta_true
        # This simulates Q^m A |0>
        # Note: In a real AE, we don't know theta_true, we implement Q operator.
        # Here we simulate the effect of Q^m on the amplitude for demonstration.
        # In a real circuit, A loads the data (p), and Q amplifies it.
        qc.ry(2 * (2 * m + 1) * theta_true, 0)
        qc.measure(0, 0)
        
        # Transpile for the backend
        qc_transpiled = transpile(qc, backend)
        
        job = backend.run(qc_transpiled, shots=shots)
        counts = job.result().get_counts()
        h = counts.get('1', 0) # Hits
        results.append((m, h, shots))
        print(f"  m={m}: hits={h}/{shots}")
        
    # Maximum Likelihood Estimation
    def likelihood(theta):
        # theta is the parameter to estimate (target is theta_true)
        # P(h|theta) = Binomial(h; N, p_m(theta))
        # p_m(theta) = sin^2((2m+1)*theta)
        log_L = 0
        for m, h, N in results:
            p_m = np.sin((2 * m + 1) * theta)**2
            # Avoid log(0)
            p_m = np.clip(p_m, 1e-9, 1 - 1e-9)
            log_L += h * np.log(p_m) + (N - h) * np.log(1 - p_m)
        return -log_L # Minimize negative log likelihood

    # Search for theta in [0, pi/2]
    res = minimize(likelihood, x0=0.1, bounds=[(0, np.pi/2)])
    theta_est = res.x[0]
    p_est = np.sin(theta_est)**2
    
    return p_est, sum(r[2] for r in results) # Return est and total shots

def main():
    print("=== Advanced Quantum Amplitude Estimation (MLAE) for Defect Rate ===")
    
    # 1. Get Risk Map from FNO
    risk_map = run_fno_inference(grid_size=64)
    threshold = 0.7
    binary_map = risk_map > threshold
    p_true = np.mean(binary_map)
    
    print(f"True Defect Ratio (Classical): {p_true:.6f}")
    
    if p_true == 0 or p_true == 1:
        print("Warning: p is 0 or 1, adjusting slightly for demonstration...")
        p_true = 0.05
    
    # 2. Run MLAE
    print("\nRunning Maximum Likelihood Amplitude Estimation (MLAE)...")
    # Schedules: Powers of 2 are common for AE
    schedules = [0, 1, 2, 4, 8] 
    shots_per_schedule = 100 # Low shots per circuit to demonstrate amplification benefit
    
    p_est, total_shots = run_mlae_estimation(p_true, m_schedules=schedules, shots=shots_per_schedule)
    
    print(f"\n[Result]")
    print(f"True p:      {p_true:.6f}")
    print(f"Estimated p: {p_est:.6f}")
    print(f"Error:       {abs(p_true - p_est):.6f}")
    print(f"Total Shots: {total_shots}")
    
    # 3. Compare with Classical Monte Carlo (Random Sampling)
    print("\n[Comparison] Classical Monte Carlo Sampling")
    # Using same total number of samples
    n_samples = total_shots
    
    # Simulate Bernoulli trials
    # We simulate measuring the original distribution (without amplification) n_samples times
    classical_hits = np.random.binomial(n_samples, p_true)
    p_classical = classical_hits / n_samples
    
    print(f"Classical p: {p_classical:.6f}")
    print(f"Error:       {abs(p_true - p_classical):.6f}")
    
    # Improvement
    error_ratio = abs(p_true - p_classical) / (abs(p_true - p_est) + 1e-9)
    print(f"\nQuantum Advantage (Error Ratio Classical/Quantum): {error_ratio:.2f}x")

if __name__ == "__main__":
    main()
