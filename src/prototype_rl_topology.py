# -*- coding: utf-8 -*-
"""
RL × Topology Optimization Prototype

MBB 梁トポロジー最適化を強化学習 (PPO) で解くプロトタイプ。
Gymnasium 環境 + stable-baselines3 PPO。

OC (Optimality Criteria) ベースラインとの比較機能付き。

Usage:
    # OC ベースライン実行
    python src/prototype_rl_topology.py --baseline

    # RL 学習
    python src/prototype_rl_topology.py --train --timesteps 100000

    # 学習済みモデルで推論
    python src/prototype_rl_topology.py --eval --model results/rl_topo_ppo.zip
"""

import argparse
import sys
import os
import numpy as np

import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium not found. Install with: pip install gymnasium")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("ERROR: stable-baselines3 not found. Install with: pip install stable-baselines3")
    sys.exit(1)

# Import our FEM solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_solver_2d import (solve_fem, density_filter, topology_optimization_oc,
                            plot_topology)


# ==============================================================================
# Gymnasium Environment
# ==============================================================================

class TopologyOptEnv(gym.Env):
    """Gymnasium environment for 2D topology optimization (MBB beam).

    Observation: density field (nelx*nely) + current compliance + volume fraction
    Action (Box): density change for each element, clipped to [-move, +move]
    Reward: negative normalized compliance + volume penalty
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, nelx: int = 30, nely: int = 10, volfrac: float = 0.5,
                 penal: float = 3.0, rmin: float = 1.5, max_steps: int = 100,
                 move: float = 0.1, render_mode=None):
        super().__init__()

        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.max_steps = max_steps
        self.move = move
        self.render_mode = render_mode

        n_elements = nelx * nely

        # Action: density change per element (continuous)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_elements,), dtype=np.float32
        )

        # Observation: density field + compliance + volume fraction
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(n_elements + 2,), dtype=np.float32
        )

        # Density filter
        self.H, self.Hs = density_filter(nelx, nely, rmin)

        # Initial compliance (for normalization)
        self._initial_compliance = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start from uniform density
        self.density = np.full((self.nelx, self.nely), self.volfrac)
        self.step_count = 0

        # Compute initial compliance
        compliance, _, _ = solve_fem(self.nelx, self.nely, self.density, self.penal)
        self._initial_compliance = compliance
        self._prev_compliance = compliance

        obs = self._get_obs(compliance)
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # Scale action from [-1,1] to [-move, +move]
        delta = action.reshape(self.nelx, self.nely) * self.move

        # Apply density filter to the change
        delta_flat = delta.flatten()
        delta_filtered = np.array(self.H.dot(delta_flat) / self.Hs)
        delta = delta_filtered.reshape(self.nelx, self.nely)

        # Update density with clipping
        self.density = np.clip(self.density + delta, 0.001, 1.0)

        # FEM solve
        compliance, _, _ = solve_fem(self.nelx, self.nely, self.density, self.penal)

        # Volume fraction
        vf = self.density.sum() / (self.nelx * self.nely)

        # Reward design:
        # 1. Compliance reduction (normalized)
        # 2. Volume constraint penalty
        # 3. Small bonus for convergence (small changes)
        compliance_reward = (self._prev_compliance - compliance) / self._initial_compliance
        volume_penalty = -2.0 * max(0.0, abs(vf - self.volfrac) - 0.02)
        reward = compliance_reward + volume_penalty

        self._prev_compliance = compliance

        # Termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs(compliance)
        info = {
            "compliance": compliance,
            "volume_fraction": vf,
            "step": self.step_count,
            "normalized_compliance": compliance / self._initial_compliance,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self, compliance: float) -> np.ndarray:
        vf = self.density.sum() / (self.nelx * self.nely)
        obs = np.concatenate([
            self.density.flatten(),
            [compliance / self._initial_compliance, vf]
        ]).astype(np.float32)
        return obs

    def render(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            plt.imshow(-self.density.T, cmap='gray', interpolation='none')
            plt.title(f"Step {self.step_count}")
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(-self.density.T, cmap='gray', interpolation='none')
            ax.set_title(f"Step {self.step_count}")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return img


# ==============================================================================
# Callbacks
# ==============================================================================

class TopologyCallback(BaseCallback):
    """Log topology optimization metrics during training."""

    def __init__(self, log_interval: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.best_compliance = float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            infos = self.locals.get("infos", [])
            if infos and "compliance" in infos[-1]:
                c = infos[-1]["compliance"]
                vf = infos[-1]["volume_fraction"]
                nc = infos[-1]["normalized_compliance"]
                if c < self.best_compliance:
                    self.best_compliance = c
                print(f"  Step {self.n_calls}: C={c:.4f}, Vf={vf:.4f}, "
                      f"C/C0={nc:.4f}, Best={self.best_compliance:.4f}")
        return True


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def train_rl(nelx=30, nely=10, volfrac=0.5, total_timesteps=50000,
             save_path="results/rl_topo_ppo"):
    """Train PPO agent for topology optimization."""
    print("=" * 60)
    print("RL Topology Optimization: PPO Training")
    print(f"  Grid: {nelx}x{nely}, Vf target: {volfrac}")
    print(f"  Total timesteps: {total_timesteps}")
    print("=" * 60)

    env = TopologyOptEnv(nelx=nelx, nely=nely, volfrac=volfrac, max_steps=100)

    # PPO with MLP policy (density field is relatively small)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="auto",
    )

    callback = TopologyCallback(log_interval=2000)

    print("\nTraining...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # Final evaluation
    evaluate_rl(model, env, save_path)

    return model


def evaluate_rl(model, env=None, save_prefix="results/rl_topo"):
    """Evaluate trained RL agent and compare with OC baseline."""
    if env is None:
        env = TopologyOptEnv(nelx=30, nely=10, volfrac=0.5, max_steps=100)

    print("\n" + "=" * 60)
    print("Evaluation: RL Agent")
    print("=" * 60)

    obs, _ = env.reset()
    total_reward = 0
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            print(f"  Step {step}: C={info['compliance']:.4f}, "
                  f"Vf={info['volume_fraction']:.4f}")
        if terminated or truncated:
            break

    print(f"\nRL Result: C={info['compliance']:.4f}, Vf={info['volume_fraction']:.4f}")
    print(f"  Total reward: {total_reward:.4f}")

    # Save RL result
    os.makedirs("results", exist_ok=True)
    plot_topology(env.density, f"RL (PPO): C={info['compliance']:.4f}, Vf={info['volume_fraction']:.4f}",
                  save_path=f"{save_prefix}_result.png")

    return env.density, info


def run_comparison(nelx=30, nely=10, volfrac=0.5, total_timesteps=50000):
    """Run both OC baseline and RL, then compare."""
    print("\n" + "=" * 60)
    print("COMPARISON: OC Baseline vs RL (PPO)")
    print("=" * 60)

    # 1. OC Baseline
    print("\n--- OC Baseline ---")
    oc_density, oc_history = topology_optimization_oc(
        nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=1.5
    )
    oc_final = oc_history[-1]
    os.makedirs("results", exist_ok=True)
    plot_topology(oc_density, f"OC: C={oc_final[1]:.4f} ({oc_final[0]+1} iters)",
                  save_path="results/comparison_oc.png")

    # 2. RL
    print("\n--- RL (PPO) ---")
    model = train_rl(nelx=nelx, nely=nely, volfrac=volfrac,
                     total_timesteps=total_timesteps,
                     save_path="results/rl_topo_ppo")

    env = TopologyOptEnv(nelx=nelx, nely=nely, volfrac=volfrac, max_steps=100)
    rl_density, rl_info = evaluate_rl(model, env, save_prefix="results/comparison_rl")

    # 3. Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  OC: Compliance = {oc_final[1]:.4f}, Vf = {oc_final[2]:.4f}, "
          f"Iters = {oc_final[0]+1}")
    print(f"  RL: Compliance = {rl_info['compliance']:.4f}, "
          f"Vf = {rl_info['volume_fraction']:.4f}")
    ratio = rl_info['compliance'] / oc_final[1]
    print(f"  RL/OC ratio: {ratio:.4f} (1.0 = same, <1.0 = RL better)")
    print("=" * 60)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL × Topology Optimization (MBB Beam)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run OC baseline only")
    parser.add_argument("--train", action="store_true",
                        help="Train PPO agent")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate trained model")
    parser.add_argument("--compare", action="store_true",
                        help="Run OC vs RL comparison")
    parser.add_argument("--model", default="results/rl_topo_ppo",
                        help="Path to saved model for --eval")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps")
    parser.add_argument("--nelx", type=int, default=30,
                        help="Elements in x direction")
    parser.add_argument("--nely", type=int, default=10,
                        help="Elements in y direction")
    parser.add_argument("--volfrac", type=float, default=0.5,
                        help="Target volume fraction")
    args = parser.parse_args()

    if args.baseline:
        density, history = topology_optimization_oc(
            nelx=args.nelx, nely=args.nely, volfrac=args.volfrac
        )
        os.makedirs("results", exist_ok=True)
        plot_topology(density,
                      f"OC Baseline ({args.nelx}x{args.nely}, vf={args.volfrac})",
                      save_path="results/oc_baseline_mbb.png")

    elif args.train:
        train_rl(nelx=args.nelx, nely=args.nely, volfrac=args.volfrac,
                 total_timesteps=args.timesteps)

    elif args.eval:
        env = TopologyOptEnv(nelx=args.nelx, nely=args.nely, volfrac=args.volfrac)
        model = PPO.load(args.model)
        evaluate_rl(model, env)

    elif args.compare:
        run_comparison(nelx=args.nelx, nely=args.nely, volfrac=args.volfrac,
                       total_timesteps=args.timesteps)

    else:
        # Default: run comparison
        run_comparison(nelx=args.nelx, nely=args.nely, volfrac=args.volfrac,
                       total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
