import gymnasium as gym
from envs.trace_env import TraceEnv
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import os

# ----------------------------
# Simple Policy Network
# ----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

# ----------------------------
# Training loop
# ----------------------------
def train_agent(episodes=1000, checkpoint_interval=100):
    env = TraceEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_size, n_actions)
    optimizer = Adam(policy.parameters(), lr=1e-3)

    # Load checkpoint if exists
    checkpoint_path = 'agents/policy_model.pth'
    if os.path.exists(checkpoint_path):
        policy.load_state_dict(torch.load(checkpoint_path))
        print(f"✅ Loaded checkpoint from {checkpoint_path}")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_tensor)
            action = torch.multinomial(probs, 1).item()
            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Policy gradient update
            loss = -torch.log(probs[action]) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}, total reward: {total_reward}")

        if (ep + 1) % checkpoint_interval == 0:
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint saved at episode {ep+1}")

    # Save final model
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"✅ Training complete. Model saved as {checkpoint_path}")

if __name__ == "__main__":
    train_agent(episodes=1000, checkpoint_interval=100)
