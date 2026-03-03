import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
import time

# Try to import Sampler
# We prefer the reference Sampler for stability in this demo without needing explicit transpilation handling
try:
    # Qiskit 1.0+ preferred
    from qiskit.primitives import StatevectorSampler as Sampler
    print("Using StatevectorSampler (Qiskit 1.0+)")
except ImportError:
    try:
        from qiskit.primitives import Sampler
        print("Using Reference Sampler")
    except ImportError:
        # Fallback to Aer if available
        from qiskit_aer.primitives import Sampler
        print("Using Aer Sampler")

def prepare_data():
    # Simulate defect data (non-linear boundary)
    X, y = make_moons(n_samples=50, noise=0.1, random_state=42)
    
    # Rescale to [0, 2pi] for quantum encoding
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    print("=== Quantum Machine Learning (QML) Defect Classifier ===")
    
    # 1. Prepare Data
    X_train, X_test, y_train, y_test = prepare_data()
    
    num_qubits = 2
    print(f"Data shape: {X_train.shape}, Qubits: {num_qubits}")
    
    # 2. Define Feature Map and Ansatz
    # ZZFeatureMap is suitable for data encoding
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
    
    # RealAmplitudes is a hardware-efficient ansatz
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement='linear')
    
    # 3. Create VQC
    # Note: VQC automatically combines feature_map and ansatz
    sampler = Sampler()
    
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=50),
        sampler=sampler
    )
    
    # 4. Train
    print("\nTraining VQC model...")
    start_time = time.time()
    
    # Fit
    # Note: In Qiskit 1.0, VQC might need checking if it supports the new Sampler
    # If using StatevectorSampler, it should be fine.
    try:
        vqc.fit(X_train, y_train)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Attempting to use a different optimizer or configuration...")
        return
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # 5. Evaluate
    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)
    
    print(f"\n[Results]")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy:  {test_score:.4f}")
    
    if test_score > 0.7:
        print("SUCCESS: QML model achieved good accuracy!")
    else:
        print("Note: Accuracy might be low due to limited iterations or data.")

if __name__ == "__main__":
    main()
