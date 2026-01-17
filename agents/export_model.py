# agents/export_model.py
import torch
from agents.train_agent import PolicyNet

# Adjust obs_size & n_actions to match your env settings
# Use the same obs_size as env.observation_space.shape[0]
OBS_SIZE = 8 * 5   # replace with actual if different (n_events * history_len)
N_ACTIONS = 2

# Create model architecture and load weights
policy = PolicyNet(OBS_SIZE, N_ACTIONS)
policy.load_state_dict(torch.load('agents/policy_model.pth', map_location='cpu'))
policy.eval()

# Example input (batch of 1)
example_input = torch.randn(1, OBS_SIZE)

# TorchScript
traced = torch.jit.trace(policy, example_input)
traced.save('agents/policy_model_ts.pt')
print("Saved TorchScript model: agents/policy_model_ts.pt")

# Also optionally save ONNX (for use with ONNX Runtime if desired)
torch.onnx.export(policy, example_input, 'agents/policy_model.onnx',
                  input_names=['input'], output_names=['probs'],
                  dynamic_axes={'input': {0: 'batch'}, 'probs': {0: 'batch'}},
                  opset_version=11)
print("Saved ONNX model: agents/policy_model.onnx")
