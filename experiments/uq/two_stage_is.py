import argparse
import json
import numpy as np

def two_stage_importance_sampling(coarse_scores, fine_eval, k, t_coarse, t_final):
    n = len(coarse_scores)
    idx = np.argsort(np.abs(coarse_scores - t_coarse))[:k]
    vals = []
    for i in idx:
        vals.append(fine_eval(i))
    vals = np.array(vals)
    p_hat = float((vals >= t_final).mean())
    return {'selected': idx.tolist(), 'p_hat': p_hat, 'k': int(k)}

def demo():
    parser = argparse.ArgumentParser(description='Two-stage IS PoC')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--t_coarse', type=float, default=0.5)
    parser.add_argument('--t_final', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='experiments/uq/results_is.json')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.2, size=args.n)
    coarse = x + rng.normal(0, args.noise, size=args.n)
    def fine_eval(i):
        return x[i]
    res = two_stage_importance_sampling(coarse, fine_eval, args.k, args.t_coarse, args.t_final)
    with open(args.output, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    demo()
