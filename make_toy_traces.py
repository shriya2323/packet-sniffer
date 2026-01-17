import pickle
import random

# ===============================
# Configuration
# ===============================
NUM_TRACES = 300              # total traces
MIN_TRACE_LEN = 50
MAX_TRACE_LEN = 150
MALICIOUS_RATIO = 0.35        # fraction of traces that are malicious
MIN_ATTACK_LEN = 5
MAX_ATTACK_LEN = 25

EVENTS = ['read_file', 'write_file', 'open_socket', 'close_socket',
          'create_process', 'terminate_process', 'load_library', 'delete_file']

# ===============================
# Helper functions
# ===============================
def generate_trace(trace_len, malicious=False):
    trace = []
    for i in range(trace_len):
        event = random.choice(EVENTS)
        trace.append(event)
    if malicious:
        attack_len = random.randint(MIN_ATTACK_LEN, MAX_ATTACK_LEN)
        # Insert an "attack sequence" at a random position
        attack_sequence = ['create_process', 'load_library', 'open_socket'] * (attack_len // 3)
        pos = random.randint(0, trace_len - len(attack_sequence))
        trace[pos:pos+len(attack_sequence)] = attack_sequence
    return trace

# ===============================
# Generate traces
# ===============================
traces = []
num_malicious = int(NUM_TRACES * MALICIOUS_RATIO)
num_benign = NUM_TRACES - num_malicious

# Benign traces
for _ in range(num_benign):
    trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
    trace = generate_trace(trace_len, malicious=False)
    traces.append({'events': trace, 'meta': {'malicious': False}})

# Malicious traces
for _ in range(num_malicious):
    trace_len = random.randint(MIN_TRACE_LEN, MAX_TRACE_LEN)
    trace = generate_trace(trace_len, malicious=True)
    traces.append({'events': trace, 'meta': {'malicious': True}})

# Shuffle all traces
random.shuffle(traces)

# ===============================
# Save traces
# ===============================
with open('data/toy_traces.pkl', 'wb') as f:
    pickle.dump(traces, f)

# ===============================
# Summary
# ===============================
malicious_count = sum(1 for t in traces if t['meta']['malicious'])
trace_lengths = [len(t['events']) for t in traces]
attack_lengths = [len(t['events']) for t in traces if t['meta']['malicious']]

print(f"Wrote data/toy_traces.pkl: {NUM_TRACES} traces, {malicious_count} malicious ({MALICIOUS_RATIO*100:.2f}%)")
print(f"Trace length min/max/mean = {min(trace_lengths)}/{max(trace_lengths)}/{sum(trace_lengths)/len(trace_lengths):.1f}")
print(f"Attack length min/max/mean = {min(attack_lengths)}/{max(attack_lengths)}/{sum(attack_lengths)/len(attack_lengths):.1f}")

