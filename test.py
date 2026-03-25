"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from src.env import create_train_env
from src.model import ActorCritic
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000, help="Max steps per episode.")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save eval_results.csv. Skipped if not provided.")
    parser.add_argument("--gif-path", type=str, default="eval_best.gif",
                        help="Output GIF path for the best trajectory.")
    parser.add_argument("--fps", type=int, default=30, help="GIF FPS.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy distribution instead of argmax.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()
    return args


def render_rgb(env, obs_fallback):
    try:
        frame = env.render(mode="rgb_array")
        if frame is not None:
            return np.array(frame, copy=True)
    except TypeError:
        pass
    except Exception:
        pass
    # fallback: last channel of stacked obs (1, C, H, W) or (C, H, W)
    fallback = np.array(obs_fallback).squeeze()
    if fallback.ndim == 3:
        gray = fallback[-1]  # last frame in stack
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(gray_u8[..., None], 3, axis=2)
    return np.zeros((84, 84, 3), dtype=np.uint8)


def run_episode(env, model, device, max_steps, stochastic=False):
    raw_state = env.reset()
    frames = [render_rgb(env, raw_state)]
    state = torch.from_numpy(raw_state)
    total_return = 0.0
    info = {}
    success = False

    for _ in range(max_steps):
        if torch.cuda.is_available():
            state = state.cuda()
        with torch.no_grad():
            logits, _ = model(state)
            if stochastic:
                action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                action = torch.argmax(F.softmax(logits, dim=1)).item()

        raw_state, reward, done, info = env.step(action)
        state = torch.from_numpy(raw_state)
        frames.append(render_rgb(env, raw_state))
        total_return += float(reward)

        if info.get("flag_get", False):
            success = True
            break
        if done:
            break

    return total_return, frames, info, success


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)

    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    env = create_train_env(opt.world, opt.stage, actions, None)
    ckpt_path = opt.checkpoint
    model = ActorCritic(env.observation_space.shape[0], len(actions))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(ckpt_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()

    episode_returns = []
    episode_x_positions = []
    episode_successes = []
    episode_stages = []
    episode_time_used = []
    per_episode_rows = []
    best_frames = None
    best_x_pos = -1

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Running {opt.episodes} trajectories...")

    for ep in tqdm(range(1, opt.episodes + 1), desc="Running episodes"):
        ep_return, ep_frames, ep_info, ep_success = run_episode(
            env=env, model=model, device="cuda" if torch.cuda.is_available() else "cpu",
            max_steps=opt.max_steps, stochastic=opt.stochastic,
        )
        ep_x_pos = ep_info.get("x_pos", 0)
        ep_stage = ep_info.get("stage", opt.stage)
        ep_time_used = 400 - ep_info.get("time", 400) if ep_success else None
        episode_returns.append(ep_return)
        episode_x_positions.append(ep_x_pos)
        episode_successes.append(int(ep_success))
        episode_stages.append(ep_stage)
        if ep_time_used is not None:
            episode_time_used.append(ep_time_used)
        per_episode_rows.append({
            "episode": ep, "return": ep_return,
            "x_pos": ep_x_pos, "stage": ep_stage, "flag_get": int(ep_success),
            "time_used": ep_time_used,
        })
        time_str = f" time_used={ep_time_used}s" if ep_time_used is not None else ""
        print(f"Episode {ep:02d}: return={ep_return:.2f} x_pos={ep_x_pos} "
              f"flag_get={ep_info.get('flag_get', False)} stage={ep_stage}{time_str}")
        if ep_x_pos > best_x_pos:
            best_x_pos = ep_x_pos
            best_frames = ep_frames

    avg_return = float(np.mean(episode_returns))
    avg_x_pos = float(np.mean(episode_x_positions))
    max_x_pos = int(max(episode_x_positions))
    max_stage = int(max(episode_stages))
    success_rate = float(np.mean(episode_successes))
    avg_time_used = float(np.mean(episode_time_used)) if episode_time_used else None

    print(f"\n--- Summary over {opt.episodes} trajectories ---")
    print(f"avg_return   : {avg_return:.2f}")
    print(f"avg_x_pos    : {avg_x_pos:.1f}")
    print(f"max_x_pos    : {max_x_pos}")
    print(f"max_stage    : {max_stage}")
    print(f"success_rate : {success_rate:.2%}  ({sum(episode_successes)}/{opt.episodes})")
    if avg_time_used is not None:
        print(f"avg_time_used: {avg_time_used:.1f}s  (over {len(episode_time_used)} successful episodes)")

    gif_out = Path(opt.log_dir) / opt.gif_path if opt.log_dir is not None else Path(opt.gif_path)
    gif_out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_out), best_frames, fps=opt.fps)
    print(f"\nSaved GIF : {gif_out} (furthest x_pos={best_x_pos})")

    if opt.log_dir is not None:
        log_dir = Path(opt.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "eval_results.csv"
        timestamp = datetime.now().isoformat()
        fields = ["timestamp", "checkpoint", "avg_return", "avg_x_pos", "max_x_pos", "max_stage",
                  "success_rate", "avg_time_used", "episode", "return", "x_pos", "stage", "flag_get", "time_used"]
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for row in per_episode_rows:
                writer.writerow({
                    "timestamp": timestamp, "checkpoint": ckpt_path,
                    "avg_return": avg_return, "avg_x_pos": avg_x_pos,
                    "max_x_pos": max_x_pos, "max_stage": max_stage,
                    "success_rate": success_rate, "avg_time_used": avg_time_used,
                    **row,
                })
        print(f"Logged results : {csv_path}")

        json_path = log_dir / "eval_summary.json"
        summary = {
            "timestamp": timestamp,
            "checkpoint": ckpt_path,
            "episodes": opt.episodes,
            "seed": opt.seed,
            "stochastic": opt.stochastic,
            "avg_return": avg_return,
            "avg_x_pos": avg_x_pos,
            "max_x_pos": max_x_pos,
            "max_stage": max_stage,
            "success_rate": success_rate,
            "avg_time_used": avg_time_used,
            "per_episode": per_episode_rows,
        }
        class _Encoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                return super().default(o)

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, cls=_Encoder)
        print(f"JSON summary   : {json_path}")

    env.close()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
