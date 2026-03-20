import sys
import time
import numpy as np  # ← Missing import added
import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


class ActorCritic(nn.Module):
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
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # Handle different input shapes: (H,W,C), (B,H,W,C), or (B,C,H,W)
        if x.dim() == 4:
            # Could be (B,H,W,C) → permute to (B,C,H,W)
            if x.shape[-1] == 4:  # likely (B,H,W,4)
                x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            # (H,W,C) → (C,H,W)
            x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)  # add batch dim → (1,C,H,W)

        x = self.net(x)
        x = x.reshape(-1, 20736)
        x = torch.relu(self.linear(x))

        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act_greedy(self, obs):
        """Greedy action selection for evaluation"""
        logits, value = self.forward(obs)
        action = torch.argmax(logits, dim=-1)
        return action, value


def preprocess_obs(obs):
    """
    Convert environment observation to model input tensor.
    Assumes obs is (H, W, C) numpy array from wrap_mario.
    Returns (1, C, H, W) torch tensor on correct device.
    """
    if isinstance(obs, tuple):  # gymnasium sometimes returns (obs, info)
        obs = obs[0]
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)

    assert len(obs.shape) == 3, f"Expected (H,W,C), got {obs.shape}"

    # (H,W,C) → (C,H,W) → (1,C,H,W)
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)

    return torch.tensor(obs, dtype=torch.float32)


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "mario_q_target.pth"
    print(f"Load ckpt from {ckpt_path}")

    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Fix: use .observation_space (not .single_observation_space) for single env
    obs_shape = env.observation_space.shape  # e.g., (240, 256, 4)
    act_dim = env.action_space.n

    model = ActorCritic(n_frame, act_dim).to(device)
    model.load_state_dict(torch.load(
        ckpt_path,
        map_location=torch.device(device)),
        strict=False # Important for NNs with different heads :)
    )
    model.eval()

    total_score = 0.0
    done = False
    truncated = False

    # Reset and preprocess initial observation
    reset_result = env.reset()
    if isinstance(reset_result, tuple):  # gymnasium API
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    s = preprocess_obs(obs).to(device)

    print("Starting evaluation...")
    step = 0
    while not (done or truncated):
        env.render(mode='human')  # ← Explicit render mode

        with torch.no_grad():
            # ← Use greedy action, not sampling, for consistent eval
            action, value = model.act_greedy(s)
            action_idx = action.item()

        # Step environment (handle gym/gymnasium API differences)
        step_result = env.step(action_idx)
        if len(step_result) == 4:
            # Older gym: obs, reward, done, info
            obs_prime, reward, done, info = step_result
            truncated = False
        else:
            # gymnasium: obs, reward, terminated, truncated, info
            obs_prime, reward, done, truncated, info = step_result

        total_score += reward
        s = preprocess_obs(obs_prime).to(device)
        step += 1

        # Optional: small delay for visibility, can be removed for speed
        time.sleep(0.01)

    stage = getattr(env.unwrapped, '_stage', info.get('stage', -1))
    print(f"\nEpisode finished after {step} steps")
    print(f"Total score : {total_score:.2f} | Stage : {stage}")
    print(f"Final info: {info}")

    env.close()