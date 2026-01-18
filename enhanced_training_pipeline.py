#!/usr/bin/env python3
"""
Enhanced Data Generation and Training Pipeline for Packet Sniffer RL
This script generates improved training data and trains a model with better accuracy.
"""

import pickle
import random
import gymnasium as gym
from envs.trace_env import TraceEnv
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import os
from datetime import datetime

# ===============================
# ENHANCED CONFIGURATION
# ===============================
NUM_TRACES_TRAIN = 1000           # increased from 300 for better training
NUM_TRACES_TEST = 200             # separate test set
MIN_TRACE_LEN = 30                # reduced minimum for more varied sequences
MAX_TRACE_LEN = 200               # increased maximum for longer patterns
MALICIOUS_RATIO = 0.4             # balanced ratio
MIN_ATTACK_LEN = 8                # longer attack sequences
MAX_ATTACK_LEN = 40               # longer attack sequences

# Enhanced event vocabulary to better represent network behavior
ENHANCED_EVENTS = [
    # File operations
    'read_file', 'write_file', 'delete_file', 'create_file', 'rename_file',
    # Network operations  
    'open_socket', 'close_socket', 'connect_to_host', 'send_data', 'receive_data',
    # Process operations
    'create_process', 'terminate_process', 'load_library', 'unload_library',
    # System operations
    'modify_registry', 'access_system_api', 'allocate_memory', 'free_memory',
    # Security-related
    'access_password', 'modify_permissions', 'scan_network', 'encrypt_data'
]

# Attack patterns for more realistic malicious behavior
ATTACK_PATTERNS = [
    # Network scanning pattern
    ['scan_network', 'connect_to_host', 'send_data', 'receive_data'] * 3,
    # File manipulation pattern  
    ['create_file', 'write_file', 'modify_permissions', 'encrypt_data'] * 2,
    # Process injection pattern
    ['create_process', 'load_library', 'access_system_api', 'modify_registry'] * 2,
    # Combined attack
    ['scan_network', 'connect_to_host', 'create_process', 'load_library', 'send_data', 'receive_data'] * 2
]

# ===============================
# ENHANCED DATA GENERATION
# ===============================
def generate_enhanced_trace(trace_len, malicious=False):
    """Generate a trace with enhanced realism."""
    trace = []
    
    for i in range(trace_len):
        # Base random event
        event = random.choice(ENHANCED_EVENTS)
        trace.append(event)
        
        # Occasionally add correlated events (more realistic behavior)
        if random.random() < 0.15:  # 15% chance of correlated sequence
            base_event = event
            if base_event.startswith('open_') or base_event.startswith('connect'):
                # Network correlation
                trace.extend(['send_data', 'receive_data'])
            elif base_event.startswith('create_') or base_event.startswith('load'):
                # Process correlation
                trace.extend(['allocate_memory', 'access_system_api'])
            elif base_event.startswith('read_') or base_event.startswith('write'):
                # File correlation
                trace.extend(['access_system_api', 'modify_permissions'])
            
            # Trim to original length if needed
            if len(trace) > trace_len:
                trace = trace[:trace_len]
                break
    
    if malicious:
        # Select a random attack pattern
        attack_pattern = random.choice(ATTACK_PATTERNS)
        # Insert attack sequence at a random position
        attack_len = min(len(attack_pattern), MAX_ATTACK_LEN)
        pos = random.randint(0, max(0, trace_len - attack_len))
        trace[pos:pos+attack_len] = attack_pattern[:attack_len]
        
        # Ensure trace is correct length after insertion
        if len(trace) > trace_len:
            trace = trace[:trace_len]
        elif len(trace) < trace_len:
            # Fill remaining slots with random events
            while len(trace) < trace_len:
                trace.append(random.choice(ENHANCED_EVENTS))
    
    return trace

def generate_enhanced_traces():
    """Generate enhanced training and test traces."""
    print("🔄 Generating enhanced training data...")
    
    # Training traces
    train_traces = []
    num_malicious_train = int(NUM_TRACES_TRAIN * MALICIOUS_RATIO)
    num_benign_train = NUM_TRACES_TRAIN - num_malicious_train

    # Benign training traces
    for i in range(num_benign_train):
        trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
        trace = generate_enhanced_trace(trace_len, malicious=False)
        train_traces.append({'events': trace, 'meta': {'malicious': False, 'id': f'train_benign_{i}'}})
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_benign_train} benign training traces...")

    # Malicious training traces
    for i in range(num_malicious_train):
        trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
        trace = generate_enhanced_trace(trace_len, malicious=True)
        train_traces.append({'events': trace, 'meta': {'malicious': True, 'id': f'train_malicious_{i}'}})
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_malicious_train} malicious training traces...")

    # Test traces
    test_traces = []
    num_malicious_test = int(NUM_TRACES_TEST * MALICIOUS_RATIO)
    num_benign_test = NUM_TRACES_TEST - num_malicious_test

    # Benign test traces
    for i in range(num_benign_test):
        trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
        trace = generate_enhanced_trace(trace_len, malicious=False)
        test_traces.append({'events': trace, 'meta': {'malicious': False, 'id': f'test_benign_{i}'}})

    # Malicious test traces
    for i in range(num_malicious_test):
        trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
        trace = generate_enhanced_trace(trace_len, malicious=True)
        test_traces.append({'events': trace, 'meta': {'malicious': True, 'id': f'test_malicious_{i}'}})

    # Shuffle traces
    random.shuffle(train_traces)
    random.shuffle(test_traces)

    # Save traces
    with open('data/enhanced_traces_train.pkl', 'wb') as f:
        pickle.dump(train_traces, f)
    
    with open('data/enhanced_traces_test.pkl', 'wb') as f:
        pickle.dump(test_traces, f)

    # Print summary
    train_malicious_count = sum(1 for t in train_traces if t['meta']['malicious'])
    test_malicious_count = sum(1 for t in test_traces if t['meta']['malicious'])
    
    print(f"\n📊 Training Data Summary:")
    print(f"   Total traces: {len(train_traces)}")
    print(f"   Malicious: {train_malicious_count} ({train_malicious_count/len(train_traces)*100:.1f}%)")
    
    print(f"\n📊 Test Data Summary:")
    print(f"   Total traces: {len(test_traces)}")
    print(f"   Malicious: {test_malicious_count} ({test_malicious_count/len(test_traces)*100:.1f}%)")
    
    return train_traces, test_traces

# ===============================
# ENHANCED MODEL ARCHITECTURE
# ===============================
class EnhancedPolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(EnhancedPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 256),      # Increased width
            nn.ReLU(),
            nn.Dropout(0.3),              # Added dropout for regularization
            nn.Linear(256, 128),          
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# ===============================
# ENHANCED TRAINING FUNCTION WITH EPISODE TRACKING
# ===============================
def train_enhanced_model(episodes=1000, checkpoint_interval=50):
    """Train the model with enhanced data and techniques."""
    print("\n🏋️ Starting enhanced model training...")
    print(f"Training for {episodes} episodes with checkpoint every {checkpoint_interval} episodes")
    
    # Use enhanced training data
    env = TraceEnv(trace_file='data/enhanced_traces_train.pkl')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = EnhancedPolicyNet(obs_size, n_actions)
    optimizer = Adam(policy.parameters(), lr=1e-3)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    best_reward = float('-inf')
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 500:  # Limit steps to prevent infinite loops
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_tensor)
            action = torch.multinomial(probs, 1).item()
            next_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

            # Enhanced policy gradient update with entropy regularization
            log_prob = torch.log(probs[action] + 1e-8)  # Add epsilon for numerical stability
            entropy = -(probs * torch.log(probs + 1e-8)).sum()  # Encourage exploration
            
            loss = -log_prob * reward - 0.01 * entropy  # Entropy regularization
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            optimizer.step()
            obs = next_obs

        # Update learning rate
        scheduler.step()

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes}, Total reward: {total_reward:.2f}, Steps: {step_count}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model based on reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), 'agents/best_policy_model.pth')
            print(f"  🏆 New best model saved with reward: {best_reward:.2f}")

        # Periodic checkpointing
        if (ep + 1) % checkpoint_interval == 0:
            torch.save(policy.state_dict(), f'agents/policy_model_ep{ep+1}.pth')
            print(f"  ✅ Checkpoint saved at episode {ep+1}")

    # Save final model
    torch.save(policy.state_dict(), 'agents/policy_model.pth')
    print(f"\n✅ Training complete. Final model saved as agents/policy_model.pth")
    
    # Export as TorchScript model
    dummy_input = torch.randn(1, obs_size)
    traced_model = torch.jit.trace(policy, dummy_input)
    torch.jit.save(traced_model, 'agents/policy_model_ts.pt')
    print(f"✅ TorchScript model saved as agents/policy_model_ts.pt")
    
    return policy

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(model_path='agents/policy_model.pth'):
    """Evaluate the trained model on test data."""
    print("\n🔍 Evaluating model performance...")
    
    # Load test environment
    test_env = TraceEnv(trace_file='data/enhanced_traces_test.pkl')
    obs_size = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.n

    # Load model
    policy = EnhancedPolicyNet(obs_size, n_actions)
    policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    policy.eval()

    total_episodes = 100  # Evaluate on subset for efficiency
    correct_predictions = 0
    total_predictions = 0
    malicious_correct = 0
    malicious_total = 0
    benign_correct = 0
    benign_total = 0

    with torch.no_grad():
        for ep in range(total_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_predictions = 0
            correct_in_episode = 0
            
            # Get the true label for this trace
            true_label = int(test_env.current_trace['meta']['malicious'])
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = policy(obs_tensor)
                action = torch.argmax(probs).item()  # Use argmax for deterministic evaluation
                
                obs, reward, done, _, _ = test_env.step(action)
                
                # Count correct predictions in this episode
                if action == true_label:
                    correct_in_episode += 1
                episode_predictions += 1
            
            # Overall accuracy for this episode
            if episode_predictions > 0:
                accuracy = correct_in_episode / episode_predictions
                total_predictions += episode_predictions
                correct_predictions += correct_in_episode
                
                # Class-specific metrics
                if true_label == 1:  # Malicious
                    malicious_total += 1
                    if accuracy > 0.5:  # Consider majority prediction correct
                        malicious_correct += 1
                else:  # Benign
                    benign_total += 1
                    if accuracy > 0.5:
                        benign_correct += 1

            if (ep + 1) % 20 == 0:
                print(f"  Evaluated {ep+1}/{total_episodes} test episodes...")

    # Calculate metrics
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    malicious_acc = malicious_correct / malicious_total if malicious_total > 0 else 0
    benign_acc = benign_correct / benign_total if benign_total > 0 else 0
    
    print(f"\n📈 Model Evaluation Results:")
    print(f"   Overall Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"   Malicious Accuracy: {malicious_acc:.3f} ({malicious_correct}/{malicious_total})")
    print(f"   Benign Accuracy: {benign_acc:.3f} ({benign_correct}/{benign_total})")
    
    return overall_accuracy, malicious_acc, benign_acc

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    print("🚀 Enhanced Data Generation and Training Pipeline")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('agents', exist_ok=True)
    
    # Generate enhanced training and test data
    train_traces, test_traces = generate_enhanced_traces()
    
    # Train the enhanced model
    trained_model = train_enhanced_model(episodes=1000, checkpoint_interval=50)
    
    # Evaluate the model
    overall_acc, malicious_acc, benign_acc = evaluate_model()
    
    print(f"\n🎉 Training and evaluation complete!")
    print(f"   Final Model Performance:")
    print(f"   - Overall Accuracy: {overall_acc:.3f}")
    print(f"   - Malicious Detection: {malicious_acc:.3f}")
    print(f"   - Benign Classification: {benign_acc:.3f}")
    
    print(f"\n💾 Models saved:")
    print(f"   - Standard: agents/policy_model.pth")
    print(f"   - TorchScript: agents/policy_model_ts.pt")
    print(f"   - Best performing: agents/best_policy_model.pth")
    print(f"   - Training data: data/enhanced_traces_train.pkl")
    print(f"   - Test data: data/enhanced_traces_test.pkl")

if __name__ == "__main__":
    main()