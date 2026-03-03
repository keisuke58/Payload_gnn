import argparse
import json
import math
import numpy as np

def classical_probability_estimate(values, threshold):
    return float((np.array(values) >= threshold).mean())

def build_state_preparation_for_p(p):
    try:
        from qiskit import QuantumCircuit
        theta = math.asin(max(0.0, min(1.0, p)) ** 0.5)
        A = QuantumCircuit(1)
        A.ry(2 * theta, 0)
        return A
    except Exception as e:
        return None

def run_iterative_ae(p, epsilon=0.02, alpha=0.05):
    try:
        from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
        from qiskit_aer.primitives import SamplerV2 as Sampler
        A = build_state_preparation_for_p(p)
        problem = EstimationProblem(state_preparation=A, objective_qubits=[0])
        iae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler())
        result = iae.estimate(problem)
        return {
            'est_p': float(result.estimation),
            'confidence_interval': [float(result.confidence_interval[0]), float(result.confidence_interval[1])],
            'num_oracle_queries': int(result.num_oracle_queries)
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Iterative Amplitude Estimation PoC')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='experiments/quantum/ae_iterative_results.json')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    values = rng.normal(0.48, 0.12, size=args.n)
    p_classical = classical_probability_estimate(values, args.threshold)
    ae_res = run_iterative_ae(p_classical, epsilon=args.epsilon, alpha=args.alpha)
    out = {
        'threshold': args.threshold,
        'p_classical': p_classical,
        'iae': ae_res
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
