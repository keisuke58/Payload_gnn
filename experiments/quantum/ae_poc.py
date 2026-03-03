import argparse
import json
import math
import numpy as np

def classical_probability_estimate(values, threshold):
    return float((np.array(values) >= threshold).mean())

def required_samples_classical(epsilon, delta):
    return int(math.ceil((1/(2*epsilon**2))*math.log(2/delta)))

def try_qiskit_amplitude_encoding(p, shots=2048):
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        backend = Aer.get_backend('aer_simulator')
        qc = QuantumCircuit(1, 1)
        theta = math.asin(math.sqrt(max(0.0, min(1.0, p))))
        qc.ry(2*theta, 0)
        qc.measure(0, 0)
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        p_hat = counts.get('1', 0) / shots
        return {'p_hat': float(p_hat), 'shots': int(shots), 'counts': counts, 'backend': 'qiskit_aer'}
    except Exception as e:
        return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Amplitude Estimation PoC (Quantum-Ready)')
    parser.add_argument('--n', type=int, default=1000, help='Number of coarse samples')
    parser.add_argument('--threshold', type=float, default=0.5, help='Exceedance threshold')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Target absolute error')
    parser.add_argument('--delta', type=float, default=0.05, help='Failure probability')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='experiments/quantum/ae_results.json')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values = rng.normal(0.48, 0.12, size=args.n)

    p_classical = classical_probability_estimate(values, args.threshold)
    n_required = required_samples_classical(args.epsilon, args.delta)

    qiskit_res = try_qiskit_amplitude_encoding(p_classical, shots=max(1024, n_required))

    out = {
        'threshold': args.threshold,
        'p_classical': p_classical,
        'n_required_classical': n_required,
        'qiskit': qiskit_res,
        'note': 'If qiskit is available, result shows amplitude-encoded measurement of p; otherwise classical-only.'
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
