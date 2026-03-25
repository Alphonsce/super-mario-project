import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import csv
from datetime import datetime
import torch
from src.env import MultipleEnvironments
from src.model import ActorCritic
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np


class CSVLogger:
    def __init__(self, experiment_name, log_dir="logs"):
        self.experiment_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        self.train_filepath = os.path.join(self.experiment_log_dir, "train.csv")
        self.train_updates_filepath = os.path.join(self.experiment_log_dir, "train_updates.csv")
        self.train_fields = [
            "timestamp", "update", "avg_return", "policy_loss", "value_loss", "entropy", "total_loss",
        ]
        self.train_update_fields = [
            "timestamp", "update", "avg_return", "max_stage", "mean_ep_x_pos", "train_success_rate",
        ]
        if not os.path.exists(self.train_filepath):
            with open(self.train_filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.train_fields).writeheader()
        if not os.path.exists(self.train_updates_filepath):
            with open(self.train_updates_filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.train_update_fields).writeheader()

    def log_train_step(self, update, avg_return, policy_loss, value_loss, entropy, total_loss):
        row = {
            "timestamp": datetime.now().isoformat(), "update": update,
            "avg_return": avg_return, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "total_loss": total_loss,
        }
        with open(self.train_filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.train_fields).writerow(row)

    def log_train_update(self, update, avg_return, max_stage, mean_ep_x_pos, train_success_rate):
        row = {
            "timestamp": datetime.now().isoformat(), "update": update, "avg_return": avg_return,
            "max_stage": max_stage, "mean_ep_x_pos": mean_ep_x_pos, "train_success_rate": train_success_rate,
        }
        with open(self.train_updates_filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.train_update_fields).writerow(row)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--experiment", type=str, default="mario_a2c_default")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    logger = CSVLogger(opt.experiment, log_dir=opt.log_dir)
    print(f"Train log:         {logger.train_filepath}")
    print(f"Train updates log: {logger.train_updates_filepath}")
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)

    model = ActorCritic(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()

    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    while True:
        curr_episode += 1
        log_policies = []
        values = []
        rewards = []
        dones = []
        entropies = []
        ep_x_positions = []
        ep_successes = []
        ep_stages = []

        for _ in range(opt.num_local_steps):
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            m = Categorical(policy)
            action = m.sample()
            log_policies.append(m.log_prob(action))
            entropies.append(m.entropy())

            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            for i, (d, inf) in enumerate(zip(done, info)):
                ep_stages.append(inf.get("stage", 1))
                if d:
                    ep_x_positions.append(inf.get("x_pos", 0))
                    ep_successes.append(int(inf.get("flag_get", False)))
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        # Bootstrap final value for n-step TD targets
        _, next_value = model(curr_states)
        next_value = next_value.squeeze()

        values = torch.cat(values)
        log_policies = torch.cat(log_policies)
        entropies = torch.cat(entropies)

        # Compute n-step TD returns (backwards)
        R = []
        R_t = next_value.detach()
        for reward, done in list(zip(rewards, dones))[::-1]:
            R_t = reward + opt.gamma * R_t * (1 - done)
            R.append(R_t)
        R = R[::-1]
        R = torch.cat(R).detach()

        advantages = R - values.detach()

        actor_loss = -torch.mean(log_policies * advantages)
        critic_loss = F.smooth_l1_loss(values, R)
        entropy_loss = torch.mean(entropies)
        total_loss = actor_loss + critic_loss - opt.beta * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        avg_return = R.mean().item()
        max_stage = max(ep_stages) if ep_stages else 1
        mean_ep_x_pos = float(np.mean(ep_x_positions)) if ep_x_positions else float("nan")
        train_success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0

        logger.log_train_step(
            update=curr_episode, avg_return=avg_return,
            policy_loss=actor_loss.item(), value_loss=critic_loss.item(),
            entropy=entropy_loss.item(), total_loss=total_loss.item(),
        )
        logger.log_train_update(curr_episode, avg_return, max_stage, mean_ep_x_pos, train_success_rate)

        print(f"Update {curr_episode}: avg_return={avg_return:.2f} max_stage={max_stage} "
              f"mean_ep_x_pos={mean_ep_x_pos:.1f} train_success_rate={train_success_rate:.3f} "
              f"pol={actor_loss.item():.4f} val={critic_loss.item():.4f} ent={entropy_loss.item():.4f}")

        if curr_episode % opt.save_interval == 0:
            x_pos_str = f"{mean_ep_x_pos:.1f}" if not np.isnan(mean_ep_x_pos) else "nan"
            ckpt_dir = os.path.join(opt.saved_path, opt.experiment)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f"update_{curr_episode}_x{x_pos_str}.pt",
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
