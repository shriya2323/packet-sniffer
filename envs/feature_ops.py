import numpy as np

# Define possible events
EVENTS = ['read_file','write_file','open_socket','close_socket',
          'create_process','terminate_process','load_library','delete_file']
EVENT_IDX = {e:i for i,e in enumerate(EVENTS)}
N_EVENTS = len(EVENTS)
HISTORY_LEN = 10  # last 10 events

def event_to_onehot(event_name):
    v = np.zeros(N_EVENTS, dtype=np.float32)
    idx = EVENT_IDX.get(event_name)
    if idx is not None:
        v[idx] = 1.0
    return v

class SlidingWindow:
    """Maintains the last HISTORY_LEN events"""
    def __init__(self, history_len=HISTORY_LEN):
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.window = [np.zeros(N_EVENTS, dtype=np.float32) for _ in range(self.history_len)]

    def push_event(self, event_name):
        onehot = event_to_onehot(event_name)
        self.window.pop(0)
        self.window.append(onehot)

    def get_observation(self):
        return np.concatenate(self.window).astype(np.float32)
