# agents/model_loader.py
import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from envs.feature_ops import SlidingWindow

class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

def load_model(ts_path="agents/policy_model_ts.pt", pth_path="agents/policy_model.pth"):
    # instantiate obs size from sliding window
    _tmp = SlidingWindow()
    obs_size = _tmp.get_observation().shape[0]
    n_actions = 2

    model = None
    if os.path.exists(ts_path):
        try:
            model = torch.jit.load(ts_path)
            model.eval()
            print("Loaded TorchScript model:", ts_path)
            return model
        except Exception as e:
            print("Failed to load TorchScript:", e)

    # fallback to loading state dict into PolicyNet
    net = PolicyNet(obs_size, n_actions)
    if os.path.exists(pth_path):
        state = torch.load(pth_path, map_location="cpu")
        try:
            net.load_state_dict(state)
            net.eval()
            print("Loaded weights from:", pth_path)
        except Exception as e:
            print("Failed to load state dict:", e)
    else:
        print("No model file found. Server will return neutral scores.")
    return net
