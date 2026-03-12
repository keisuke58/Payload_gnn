"""
H3 Booster RL Landing Guidance — PPO (Proximal Policy Optimization)
===================================================================
Step D: Reinforcement learning for powered descent guidance.

Custom 2D environment (no Box2D dependency) with:
  - H3 1st stage booster physics (mass, thrust, gravity, drag)
  - Continuous action space: [throttle, gimbal_angle]
  - Observation: [alt, horiz, vx, vy, angle, angular_vel, mass_frac]
  - Reward: shaped for precision landing + fuel efficiency

Trains a PPO agent to learn fuel-optimal hoverslam landing.
Compares RL policy vs G-FOLD optimal trajectory.
"""
import math
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ============================================================
# H3 Landing Environment
# ============================================================
@dataclass
class H3LandingConfig:
    # Booster parameters
    m_dry: float = 20000.0
    m_fuel: float = 5000.0
    thrust_max: float = 1471e3   # single LE-9
    isp: float = 400.0
    g: float = 9.81
    # Geometry
    length: float = 37.0         # booster length (m)
    inertia_factor: float = 0.08 # I = factor * m * L^2
    gimbal_max: float = 8.0      # degrees
    # Aerodynamics
    cd: float = 1.5              # drag coefficient (broadside)
    area: float = 5.2 * 37.0    # reference area (diameter * length)
    # Environment
    dt: float = 0.05             # simulation timestep
    max_time: float = 30.0       # max episode duration
    # Initial conditions (randomized ranges)
    alt_range: tuple = (1200.0, 1800.0)
    horiz_range: tuple = (-400.0, 400.0)
    vx_range: tuple = (-15.0, 15.0)
    vy_range: tuple = (-80.0, -40.0)
    angle_range: tuple = (-0.1, 0.1)     # radians
    omega_range: tuple = (-0.05, 0.05)


class H3LandingEnv:
    """2D rocket landing environment for RL training."""

    def __init__(self, config=None):
        self.cfg = config or H3LandingConfig()
        self.obs_dim = 8
        self.act_dim = 2  # [throttle (0-1), gimbal (-1 to 1)]
        self.reset()

    def reset(self, deterministic=False):
        cfg = self.cfg
        if deterministic:
            self.x = 0.0
            self.y = 1500.0
            self.vx = 0.0
            self.vy = -60.0
            self.theta = 0.0
            self.omega = 0.0
        else:
            self.x = random.uniform(*cfg.horiz_range)
            self.y = random.uniform(*cfg.alt_range)
            self.vx = random.uniform(*cfg.vx_range)
            self.vy = random.uniform(*cfg.vy_range)
            self.theta = random.uniform(*cfg.angle_range)
            self.omega = random.uniform(*cfg.omega_range)

        self.mass = cfg.m_dry + cfg.m_fuel
        self.fuel = cfg.m_fuel
        self.t = 0.0
        self.done = False
        self.landed = False
        self.crashed = False
        return self._obs()

    def _obs(self):
        cfg = self.cfg
        # Normalized observations
        return np.array([
            self.y / 2000.0,                    # altitude (norm)
            self.x / 500.0,                      # horizontal pos
            self.vx / 50.0,                      # horizontal vel
            self.vy / 100.0,                     # vertical vel
            self.theta / 0.5,                    # angle
            self.omega / 1.0,                    # angular vel
            self.fuel / cfg.m_fuel,              # fuel fraction
            self.t / cfg.max_time,               # time fraction
        ], dtype=np.float32)

    def step(self, action):
        """action: [throttle (0-1), gimbal (-1 to 1)]"""
        cfg = self.cfg
        throttle = float(np.clip(action[0], 0, 1))
        gimbal = float(np.clip(action[1], -1, 1)) * math.radians(cfg.gimbal_max)

        # Thrust
        if self.fuel > 0 and throttle > 0.01:
            thrust = throttle * cfg.thrust_max
            dm = thrust / (cfg.isp * cfg.g) * cfg.dt
            dm = min(dm, self.fuel)
            actual_thrust = dm * cfg.isp * cfg.g / cfg.dt
            self.fuel -= dm
            self.mass -= dm
        else:
            actual_thrust = 0.0
            gimbal = 0.0

        # Thrust direction (relative to body)
        # theta = 0 means vertical, positive = tilted right
        thrust_angle = self.theta + gimbal
        Tx = actual_thrust * math.sin(thrust_angle)
        Ty = actual_thrust * math.cos(thrust_angle)

        # Torque from thrust offset
        torque = actual_thrust * math.sin(gimbal) * cfg.length * 0.4

        # Aerodynamic drag
        v_mag = math.sqrt(self.vx**2 + self.vy**2)
        if v_mag > 0.1 and self.y < 50000:
            # Simple density model
            rho = 1.225 * math.exp(-self.y / 8500) if self.y > 0 else 1.225
            # Effective area depends on angle of attack
            aoa = math.atan2(self.vx, -self.vy) - self.theta if v_mag > 1 else 0
            cd_eff = cfg.cd * (0.2 + 0.8 * abs(math.sin(aoa)))
            area_eff = cfg.area * (0.1 + 0.9 * abs(math.sin(aoa)))
            drag = 0.5 * rho * v_mag**2 * cd_eff * area_eff
            drag_x = -drag * self.vx / v_mag
            drag_y = -drag * self.vy / v_mag
        else:
            drag_x = 0
            drag_y = 0

        # Accelerations
        ax = (Tx + drag_x) / self.mass
        ay = (Ty + drag_y) / self.mass - cfg.g

        inertia = cfg.inertia_factor * self.mass * cfg.length**2
        alpha = torque / inertia

        # Integration (symplectic Euler)
        self.vx += ax * cfg.dt
        self.vy += ay * cfg.dt
        self.omega += alpha * cfg.dt
        self.x += self.vx * cfg.dt
        self.y += self.vy * cfg.dt
        self.theta += self.omega * cfg.dt
        self.t += cfg.dt

        # Check termination
        reward = 0.0

        if self.y <= 0:
            self.done = True
            speed = math.sqrt(self.vx**2 + self.vy**2)
            angle_err = abs(self.theta)
            pos_err = abs(self.x)

            if speed < 5.0 and angle_err < 0.15 and pos_err < 30.0:
                # Successful landing
                self.landed = True
                reward += 200.0
                reward -= speed * 10.0       # penalize landing speed
                reward -= pos_err * 2.0      # penalize position error
                reward -= angle_err * 50.0   # penalize tilt
                # Fuel bonus
                reward += (self.fuel / cfg.m_fuel) * 50.0
            else:
                # Crash
                self.crashed = True
                reward -= 100.0
            self.y = 0.0

        elif self.t >= cfg.max_time:
            self.done = True
            reward -= 50.0  # timeout penalty

        elif self.y > 3000:
            self.done = True
            reward -= 50.0  # flew away

        else:
            # Shaping reward
            dist = math.sqrt(self.x**2 + self.y**2)
            speed = math.sqrt(self.vx**2 + self.vy**2)

            # Approach reward
            reward -= 0.001 * dist
            reward -= 0.01 * abs(self.theta)
            reward -= 0.001 * abs(self.omega)

            # Fuel efficiency
            reward -= 0.0001 * actual_thrust / cfg.thrust_max

            # Altitude-speed correlation (should slow down as altitude decreases)
            if self.y < 500:
                desired_speed = 5.0 + (self.y / 500.0) * 50.0
                if speed > desired_speed * 1.5:
                    reward -= 0.01 * (speed - desired_speed)

        return self._obs(), reward, self.done, {
            "landed": self.landed,
            "crashed": self.crashed,
            "fuel_remaining": self.fuel,
            "time": self.t,
        }


# ============================================================
# PPO Agent
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Actor (policy)
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        # Critic (value function)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.actor_mean(h)
        # Throttle: sigmoid, gimbal: tanh
        mean = torch.cat([
            torch.sigmoid(mean[:, :1]),
            torch.tanh(mean[:, 1:]),
        ], dim=1)
        std = torch.exp(self.actor_logstd.clamp(-3, 0.5))
        value = self.critic(h)
        return mean, std, value

    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mean, std, value = self(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            # Clamp actions
            action_np = action.squeeze(0).numpy()
            action_np[0] = np.clip(action_np[0], 0, 1)
            action_np[1] = np.clip(action_np[1], -1, 1)
            return action_np, log_prob.item(), value.item()


class PPOTrainer:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=4, batch_size=128):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        return advantages, returns

    def update(self, trajectories):
        obs = torch.FloatTensor(np.array([t[0] for t in trajectories]))
        acts = torch.FloatTensor(np.array([t[1] for t in trajectories]))
        old_logprobs = torch.FloatTensor([t[2] for t in trajectories])
        rewards = [t[3] for t in trajectories]
        values = [t[4] for t in trajectories]
        dones = [t[5] for t in trajectories]

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        for _ in range(self.epochs):
            indices = np.random.permutation(len(trajectories))
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start:start + self.batch_size]

                mean, std, values_pred = self.model(obs[idx])
                dist = Normal(mean, std)
                new_logprobs = dist.log_prob(acts[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_logprobs - old_logprobs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns[idx] - values_pred.squeeze()).pow(2).mean()
                loss = actor_loss + critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss


# ============================================================
# Training Loop
# ============================================================
def train(num_episodes=3000, print_every=100):
    env = H3LandingEnv()
    trainer = PPOTrainer(env.obs_dim, env.act_dim, lr=3e-4)

    rewards_history = []
    landing_rate_history = []
    fuel_history = []
    best_reward = -float('inf')
    landing_window = deque(maxlen=100)

    print("=" * 70)
    print("  H3 Booster RL Landing — PPO Training (Curriculum)")
    print("=" * 70)
    print(f"  Episodes: {num_episodes}")
    print(f"  Observation dim: {env.obs_dim}, Action dim: {env.act_dim}")

    # Collect multiple episodes before updating (mini-batch PPO)
    batch_trajectories = []
    update_every = 8  # episodes per update

    for ep in range(num_episodes):
        # Curriculum: start easy (low alt, slow) → ramp to full difficulty
        progress = min(1.0, ep / (num_episodes * 0.6))
        env.cfg.alt_range = (200 + 1300 * progress, 400 + 1400 * progress)
        env.cfg.vy_range = (-20 - 60 * progress, -10 - 30 * progress)
        env.cfg.horiz_range = (-50 - 350 * progress, 50 + 350 * progress)
        env.cfg.vx_range = (-3 - 12 * progress, 3 + 12 * progress)

        obs = env.reset()
        ep_trajectories = []
        ep_reward = 0

        while not env.done:
            action, logprob, value = trainer.model.act(obs)
            next_obs, reward, done, info = env.step(action)

            ep_trajectories.append((obs, action, logprob, reward, value, float(done)))
            obs = next_obs
            ep_reward += reward

        batch_trajectories.extend(ep_trajectories)

        # Update policy every N episodes
        if (ep + 1) % update_every == 0 and len(batch_trajectories) > 0:
            trainer.update(batch_trajectories)
            batch_trajectories = []

        rewards_history.append(ep_reward)
        landing_window.append(1.0 if info["landed"] else 0.0)
        landing_rate = sum(landing_window) / len(landing_window) * 100
        landing_rate_history.append(landing_rate)
        fuel_history.append(info["fuel_remaining"])

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(trainer.model.state_dict(), "h3_rl_best.pt")

        if (ep + 1) % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_fuel = np.mean(fuel_history[-print_every:])
            print(f"  Ep {ep+1:>5d}: avg_reward={avg_r:>8.1f}  "
                  f"landing={landing_rate:>5.1f}%  "
                  f"fuel={avg_fuel:>5.0f}kg  "
                  f"best={best_reward:.1f}  "
                  f"diff={progress:.1f}")

    return trainer, rewards_history, landing_rate_history, fuel_history


def evaluate(trainer, num_episodes=50):
    """Evaluate trained policy deterministically."""
    env = H3LandingEnv()
    results = []

    for _ in range(num_episodes):
        obs = env.reset(deterministic=False)
        trajectory = {"x": [], "y": [], "vx": [], "vy": [],
                      "theta": [], "thrust": [], "gimbal": [], "t": []}

        while not env.done:
            action, _, _ = trainer.model.act(obs)
            obs, _, done, info = env.step(action)

            trajectory["x"].append(env.x)
            trajectory["y"].append(env.y)
            trajectory["vx"].append(env.vx)
            trajectory["vy"].append(env.vy)
            trajectory["theta"].append(env.theta)
            trajectory["thrust"].append(action[0])
            trajectory["gimbal"].append(action[1])
            trajectory["t"].append(env.t)

        results.append({
            "landed": info["landed"],
            "crashed": info["crashed"],
            "fuel_remaining": info["fuel_remaining"],
            "final_speed": math.sqrt(env.vx**2 + env.vy**2),
            "final_pos_err": abs(env.x),
            "trajectory": trajectory,
        })

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Phase 1: Training PPO agent (curriculum learning)...")
    trainer, rewards, landing_rates, fuels = train(num_episodes=800, print_every=100)

    print(f"\nPhase 2: Evaluating trained policy...")
    eval_results = evaluate(trainer, num_episodes=100)
    n_landed = sum(1 for r in eval_results if r["landed"])
    n_crashed = sum(1 for r in eval_results if r["crashed"])
    avg_fuel = np.mean([r["fuel_remaining"] for r in eval_results if r["landed"]]) if n_landed > 0 else 0
    avg_speed = np.mean([r["final_speed"] for r in eval_results if r["landed"]]) if n_landed > 0 else 0
    avg_pos = np.mean([r["final_pos_err"] for r in eval_results if r["landed"]]) if n_landed > 0 else 0

    print(f"\n=== Evaluation Results (100 episodes) ===")
    print(f"  Landing rate:     {n_landed}%")
    print(f"  Crash rate:       {n_crashed}%")
    print(f"  Avg landing speed: {avg_speed:.2f} m/s")
    print(f"  Avg position error: {avg_pos:.1f} m")
    print(f"  Avg fuel remaining: {avg_fuel:.0f} kg")

    # ============================================================
    # Plots — 6-panel figure
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Training reward curve
    ax = axes[0, 0]
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'b-', lw=1.5, label=f'Smoothed (w={window})')
    ax.plot(rewards, alpha=0.2, color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('(a) Training Reward Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Landing rate over training
    ax = axes[0, 1]
    ax.plot(landing_rates, 'g-', lw=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Landing Rate (%)')
    ax.set_title('(b) Landing Success Rate (100-ep window)')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # (c) Example trajectories (successful landings)
    ax = axes[0, 2]
    successful = [r for r in eval_results if r["landed"]]
    for i, res in enumerate(successful[:10]):
        traj = res["trajectory"]
        color = plt.cm.viridis(i / max(len(successful[:10]) - 1, 1))
        ax.plot(traj["x"], traj["y"], '-', color=color, alpha=0.7, lw=1.5)
    ax.scatter([0], [0], c='red', s=100, marker='*', zorder=10, label='Target')
    ax.set_xlabel('Horizontal (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'(c) Successful Trajectories ({n_landed}/{len(eval_results)})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Thrust profile (best landing)
    ax = axes[1, 0]
    if successful:
        best_traj = successful[0]["trajectory"]
        ax.plot(best_traj["t"], best_traj["thrust"], 'orange', lw=2, label='Throttle')
        ax.plot(best_traj["t"], best_traj["gimbal"], 'blue', lw=1.5, alpha=0.7, label='Gimbal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Input')
        ax.set_title('(d) RL Control Profile (Best Landing)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # (e) Speed vs altitude (phase portrait)
    ax = axes[1, 1]
    for i, res in enumerate(successful[:10]):
        traj = res["trajectory"]
        speed = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(traj["vx"], traj["vy"])]
        color = plt.cm.viridis(i / max(len(successful[:10]) - 1, 1))
        ax.plot(speed, traj["y"], '-', color=color, alpha=0.7, lw=1.5)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('(e) Speed-Altitude Phase Portrait')
    ax.grid(True, alpha=0.3)

    # (f) Comparison summary
    ax = axes[1, 2]
    methods = ['G-FOLD\n(Convex)', 'RL\n(PPO)']
    # G-FOLD results from Step C
    gfold_fuel = 1523
    gfold_speed = 0.0
    gfold_pos_err = 0.0

    fuel_vals = [gfold_fuel, max(0, 5000 - avg_fuel)]
    speed_vals = [gfold_speed, avg_speed]
    pos_vals = [gfold_pos_err, avg_pos]

    x_pos = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x_pos - width, [f/50 for f in fuel_vals], width,
                   label='Fuel used (kg/50)', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x_pos, speed_vals, width,
                   label='Landing speed (m/s)', color='firebrick', alpha=0.7)
    bars3 = ax.bar(x_pos + width, pos_vals, width,
                   label='Position error (m)', color='forestgreen', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_title('(f) G-FOLD vs RL Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                       f'{h:.1f}', ha='center', fontsize=7)

    plt.suptitle(
        f"H3 Booster RL Landing — PPO Agent\n"
        f"Landing rate: {n_landed}%  |  "
        f"Avg speed: {avg_speed:.2f} m/s  |  "
        f"Avg fuel used: {max(0, 5000-avg_fuel):.0f} kg",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("h3_rl_landing.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: h3_rl_landing.png")
    print(f"Model saved: h3_rl_best.pt")
