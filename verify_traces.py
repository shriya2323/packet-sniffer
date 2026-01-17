# verify_traces.py
import pickle
with open('data/toy_traces.pkl','rb') as f:
    traces = pickle.load(f)

print("Total traces:", len(traces))
print("Example 1 meta:", traces[0]['meta'])
print("First 8 events of trace 1:", traces[0]['events'][:8])
# show one malicious trace meta if exists
for i,t in enumerate(traces):
    if t['meta'].get('malicious'):
        print(f"\nFound a malicious trace at index {i}, meta:", t['meta'])
        print("First 12 events:", t['events'][:12])
        break
