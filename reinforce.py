from collections import Counter
import csv
import os
from datetime import datetime

import gym_super_mario_bros
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    return env


def get_reward(r):
    r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
    return r


class PolicyNetwork(nn.Module):
    """Policy network with optional value baseline for variance reduction."""

    def __init__(self, n_frame, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(20736, 512)
        self.policy_head = nn.Linear(512, act_dim)
        # Optional baseline network (REINFORCE with baseline)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1)
        x = self.net(x)
        x = x.reshape(-1, 20736)
        x = torch.relu(self.linear(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


def compute_returns(rewards, dones, gamma=0.99):
    """Compute Monte Carlo returns with discounting."""
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return
    return returns


def rollout_reinforce(envs, model, rollout_steps, init_obs, gamma=0.99):
    """Collect trajectories for REINFORCE update."""
    obs = init_obs
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

    for _ in range(rollout_steps):
        obs_buf.append(obs)

        with torch.no_grad():
            action, logp, value = model.act(obs)

        val_buf.append(value)
        logp_buf.append(logp)
        act_buf.append(action)

        actions = action.cpu().numpy()
        next_obs, reward, done, infos = envs.step(actions)
        reward = [get_reward(r) for r in reward]

        rew_buf.append(torch.tensor(reward, dtype=torch.float32).to(device))
        done_buf.append(torch.tensor(done, dtype=torch.float32).to(device))

        for i, d in enumerate(done):
            if d:
                print(f"Env {i} done. Resetting. (info: {infos[i]})")
                next_obs[i] = envs.envs[i].reset()

        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        max_stage = max([i["stage"] for i in infos])

    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)

    returns = compute_returns(rew_buf, done_buf, gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    logp_buf = torch.stack(logp_buf)

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": logp_buf,
        "returns": returns,
        "baselines": val_buf,
        "max_stage": max_stage,
        "last_obs": obs,
    }


def evaluate_policy(env, model, episodes=5, render=False):
    """Evaluate the learned policy."""
    model.eval()
    total_returns = []
    actions = []
    stages = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        if render:
            env.render()
        while not done:
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()
                actions.append(action)
            obs, reward, done, info = env.step(action)
            stages.append(info["stage"])
            total_reward += reward

        total_returns.append(total_reward)
        info["action_count"] = Counter(actions)

    model.train()
    return np.mean(total_returns), info, max(stages)


class CSVLogger:
    """Minimal CSV logger for REINFORCE: logs policy_loss per update."""

    def __init__(self, experiment_name, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f"{experiment_name}.csv")
        self.fields = ["timestamp", "update", "policy_loss", "avg_return", "max_stage", "eval_avg_return",
                       "eval_max_stage"]

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, update, policy_loss, avg_return, max_stage, eval_avg_return=None, eval_max_stage=None):
        row = {
            "timestamp": datetime.now().isoformat(),
            "update": update,
            "policy_loss": policy_loss,
            "avg_return": avg_return,
            "max_stage": max_stage,
            "eval_avg_return": eval_avg_return,
            "eval_max_stage": eval_max_stage,
        }
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)


def train_reinforce(experiment_name="mario_reinforce_default"):
    # Initialize logger
    logger = CSVLogger(f"mario_{experiment_name}")
    print(f"📁 Logging to: {logger.filepath}")

    num_env = 8
    envs = gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(num_env)])
    obs_dim = envs.single_observation_space.shape[-1]
    act_dim = envs.single_action_space.n
    print(f"{obs_dim=} {act_dim=}")

    model = PolicyNetwork(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

    rollout_steps = 128
    gamma = 0.99
    use_baseline = False
    baseline_coef = 0.5

    eval_env = make_env()
    eval_env.reset()
    init_obs = envs.reset()
    update = 0

    while True:
        update += 1

        batch = rollout_reinforce(envs, model, rollout_steps, init_obs, gamma)
        init_obs = batch["last_obs"]

        T, N = rollout_steps, envs.num_envs
        total_size = T * N

        obs = batch["obs"].reshape(total_size, *envs.single_observation_space.shape)
        act = batch["actions"].reshape(total_size)
        logp_old = batch["logprobs"].reshape(total_size)
        returns = batch["returns"].reshape(total_size)

        if use_baseline:
            baselines = batch["baselines"].reshape(total_size)
            advantages = returns - baselines
        else:
            advantages = returns

        # === SINGLE POLICY GRADIENT UPDATE ===
        logits, value = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act)

        policy_loss = -(logp * advantages.detach()).mean()

        if use_baseline:
            value_loss = (returns - value).pow(2).mean()
            loss = policy_loss + baseline_coef * value_loss
        else:
            loss = policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # =====================================

        # Metrics
        avg_return = returns.mean().item()
        max_stage = batch["max_stage"]
        policy_loss_val = policy_loss.item()

        # Console output
        print(f"Update {update}: policy_loss={policy_loss_val:.4f}, avg_return={avg_return:.2f}, max_stage={max_stage}")

        # Log to CSV (only policy_loss as requested, plus context)
        logger.log(update, policy_loss_val, avg_return, max_stage)

        # Evaluation and saving
        if update % 10 == 0:
            avg_score, info, eval_max_stage = evaluate_policy(
                eval_env, model, episodes=1, render=False
            )
            print(f"[Eval] Update {update}: avg_return={avg_score:.2f}, eval_max_stage={eval_max_stage}")

            # Log eval results to CSV
            logger.log_update(update, avg_return, max_stage, avg_score, eval_max_stage, info)

            if eval_max_stage > 1:
                torch.save(model.state_dict(), f"mario_{experiment_name}.pt")
                print(f"🏆 Stage 1+ cleared! Model saved to mario_{experiment_name}.pt")
                break

        if update > 0 and update % 50 == 0:
            torch.save(model.state_dict(), f"mario_{experiment_name}.pt")
            print(f"💾 Checkpoint saved at update {update}")


if __name__ == "__main__":
    import sys

    experiment_name = "mario_reinforce_default"
    if "--experiment" in sys.argv:
        idx = sys.argv.index("--experiment")
        if idx + 1 < len(sys.argv):
            experiment_name = sys.argv[idx + 1]

    train_reinforce(experiment_name=experiment_name)