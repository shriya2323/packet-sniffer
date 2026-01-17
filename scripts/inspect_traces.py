# scripts/inspect_traces.py
import pickle
from collections import Counter

with open('data/toy_traces.pkl','rb') as f:
    traces = pickle.load(f)

print("Total traces:", len(traces))
counts = Counter([t['meta']['malicious'] for t in traces])
print("Label counts:", counts)

print("\nSample trace 0:")
print("Events:", traces[0]['events'])
print("Malicious:", traces[0]['meta']['malicious'])
