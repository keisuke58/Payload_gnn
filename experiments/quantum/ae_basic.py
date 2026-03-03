import argparse
import json
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer

def classical_probability_estimate(values, threshold):
    return float((np.array(values) >= threshold).mean())

def measure_amplitude(p, shots):
    backend = Aer.get_backend('aer_simulator')
    qc = QuantumCircuit(1, 1)
    theta = math.asin(max(0.0, min(1.0, p)) ** 0.5)
    qc.ry(2 * theta, 0)
    qc.measure(0, 0)
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    p_hat = counts.get('1', 0) / shots
    return {'p_hat': float(p_hat), 'shots': int(shots), 'counts': counts}

def main():
    parser = argparse.ArgumentParser(description='Basic amplitude encoding measurement')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='experiments/quantum/ae_basic_results.json')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values = rng.normal(0.48, 0.12, size=args.n)
    p_classical = classical_probability_estimate(values, args.threshold)
    q_res = measure_amplitude(p_classical, args.shots)
    out = {'threshold': args.threshold, 'p_classical': p_classical, 'qiskit': q_res}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
