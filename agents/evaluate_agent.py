from envs.trace_env import TraceEnv
import torch
from agents.train_agent import PolicyNet
import numpy as np

def evaluate_agent(episodes=100):
    env = TraceEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_size, n_actions)
    policy.load_state_dict(torch.load('agents/policy_model.pth'))
    policy.eval()

    total_correct = 0
    mal_correct = 0
    benign_correct = 0
    mal_total = 0
    benign_total = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        correct_trace = 0
        current_label = int(env.current_trace['meta']['malicious'])
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_tensor)
            action = torch.argmax(probs).item()
            obs, reward, done, _, _ = env.step(action)
            if action == current_label:
                correct_trace += 1

        total_correct += correct_trace / len(env.current_trace['events'])
        if current_label:
            mal_total += 1
            if correct_trace / len(env.current_trace['events']) > 0.5:
                mal_correct += 1
        else:
            benign_total += 1
            if correct_trace / len(env.current_trace['events']) > 0.5:
                benign_correct += 1

    print("=== Evaluation Results ===")
    print(f"Total episodes: {episodes}")
    print(f"Overall accuracy: {total_correct/episodes*100:.2f}%")
    print(f"Malicious accuracy: {mal_correct/mal_total*100:.2f}%")
    print(f"Benign accuracy: {benign_correct/benign_total*100:.2f}%")

if __name__ == "__main__":
    evaluate_agent()
