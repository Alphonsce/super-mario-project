"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

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
            "timestamp", "update", "epoch", "batch",
            "avg_return", "policy_loss", "value_loss", "entropy", "total_loss", "clip_frac",
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

    def log_train_batch(self, update, epoch, batch, avg_return, policy_loss, value_loss, entropy, total_loss, clip_frac):
        row = {
            "timestamp": datetime.now().isoformat(), "update": update, "epoch": epoch, "batch": batch,
            "avg_return": avg_return, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "total_loss": total_loss, "clip_frac": clip_frac,
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
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16, help='Split into this many batches')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)

    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")

    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--experiment", type=str, default="mario_ppo_default")
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
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        ep_x_positions = []
        ep_successes = []
        ep_stages = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
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

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        batch_counter = 0
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_counter += 1
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                clip_mask = (ratio < 1 - opt.epsilon) | (ratio > 1 + opt.epsilon)
                clip_frac = clip_mask.float().mean().item()
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                logger.log_train_batch(
                    update=curr_episode, epoch=i, batch=batch_counter,
                    avg_return=R[batch_indices].mean().item(),
                    policy_loss=actor_loss.item(), value_loss=critic_loss.item(),
                    entropy=entropy_loss.item(), total_loss=total_loss.item(),
                    clip_frac=clip_frac,
                )
                print(f"  [Upd {curr_episode} Ep {i} B {batch_counter}] "
                      f"pol={actor_loss.item():.4f} val={critic_loss.item():.4f} "
                      f"ent={entropy_loss.item():.4f} clip={clip_frac:.3f}", end='\r')
        print(" " * 120, end='\r')

        avg_return = R.mean().item()
        max_stage = max(ep_stages) if ep_stages else 1
        mean_ep_x_pos = float(np.mean(ep_x_positions)) if ep_x_positions else float("nan")
        train_success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
        print(f"Update {curr_episode}: avg_return={avg_return:.2f} max_stage={max_stage} "
              f"mean_ep_x_pos={mean_ep_x_pos:.1f} train_success_rate={train_success_rate:.3f} "
              f"total_loss={total_loss.item():.4f}")
        logger.log_train_update(curr_episode, avg_return, max_stage, mean_ep_x_pos, train_success_rate)

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
