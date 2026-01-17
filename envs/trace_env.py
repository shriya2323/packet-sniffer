import gymnasium as gym
from gymnasium import spaces
import pickle
import numpy as np
import random

class TraceEnv(gym.Env):
    """
    Gym environment for malware behavior traces.
    Observation: one-hot vector of current event + history
    Action: 0=benign, 1=malicious
    Reward: +1 for correct classification, -1 for wrong
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, trace_file='data/toy_traces.pkl', history_len=10):
        super(TraceEnv, self).__init__()
        
        # Load traces
        with open(trace_file, 'rb') as f:
            self.traces = pickle.load(f)
        
        # Events mapping
        all_events = [
            'read_file', 'write_file', 'open_socket', 'close_socket',
            'create_process', 'terminate_process', 'load_library', 'delete_file'
        ]
        self.event2idx = {e:i for i,e in enumerate(all_events)}
        self.n_events = len(all_events)
        self.history_len = history_len

        # Gym spaces
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.n_events * history_len,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0=benign, 1=malicious

        self.current_trace = None
        self.pos = 0
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose a random trace
        self.current_trace = random.choice(self.traces)
        self.pos = 0
        # Initialize history with zeros
        self.history = [np.zeros(self.n_events) for _ in range(self.history_len)]
        return np.concatenate(self.history), {}

    def step(self, action):
        trace_len = len(self.current_trace['events'])
        done = False
        reward = 0

        # Determine correct label
        correct_label = int(self.current_trace['meta']['malicious'])
        reward = 1 if action == correct_label else -1

        # Check if we reached the end of the trace
        if self.pos >= trace_len:
            done = True
            obs = np.concatenate(self.history)
        else:
            # Move to next event
            event_name = self.current_trace['events'][self.pos]
            onehot = np.zeros(self.n_events)
            onehot[self.event2idx[event_name]] = 1
            self.history.pop(0)
            self.history.append(onehot)
            obs = np.concatenate(self.history)
            self.pos += 1

        return obs, reward, done, False, {}

    def render(self, mode='human'):
        if self.current_trace:
            print(f"Trace pos {self.pos}/{len(self.current_trace['events'])}, malicious={self.current_trace['meta']['malicious']}")
