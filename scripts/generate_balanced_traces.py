import pickle, random

events = [
    'read_file', 'write_file', 'open_socket', 'close_socket',
    'create_process', 'terminate_process', 'load_library', 'delete_file'
]

def make_trace(is_malicious):
    trace = {
        "events": random.choices(events, k=15),
        "meta": {"malicious": is_malicious}
    }
    if is_malicious:
        # Add suspicious patterns
        trace["events"] += random.choices(['delete_file', 'create_process', 'open_socket'], k=3)
    else:
        # Add benign patterns
        trace["events"] += random.choices(['read_file', 'write_file', 'load_library'], k=3)
    return trace

traces = []
for _ in range(100):  # 100 malicious + 100 benign
    traces.append(make_trace(is_malicious=True))
    traces.append(make_trace(is_malicious=False))

with open('data/toy_traces.pkl', 'wb') as f:
    pickle.dump(traces, f)

print("✅ Balanced traces saved to data/toy_traces.pkl")
