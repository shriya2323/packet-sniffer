# tools/sanity_check_model.py
import pickle, torch, numpy as np, os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.feature_ops import SlidingWindow
from collections import defaultdict

MODEL_PATH = "agents/policy_model_ts.pt"
DATA_PATH = "data/toy_traces.pkl"

def score_trace(trace, model, window_len=5):
    win = SlidingWindow(history_len=window_len)
    scores = []
    for ev in trace['events']:
        win.push_event(ev)
        obs = win.get_observation()
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(x).cpu().numpy().squeeze().tolist()
        # handle outputs
        if isinstance(p, (list, tuple)):
            score = float(p[1]) if len(p) > 1 else float(p[0])
        else:
            score = float(p)
        scores.append(score)
    return scores

def main():
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    assert os.path.exists(DATA_PATH), f"Data not found: {DATA_PATH}"

    model = torch.jit.load(MODEL_PATH)
    model.eval()

    with open(DATA_PATH, 'rb') as f:
        traces = pickle.load(f)

    # sample up to 10 malicious and 10 benign
    sample_mal = [t for t in traces if t['meta'].get('malicious')][:10]
    sample_ben = [t for t in traces if not t['meta'].get('malicious')][:10]

    def summarize(samples, label):
        all_scores = []
        for i,t in enumerate(samples):
            scores = score_trace(t, model)
            all_scores.append(np.mean(scores[-10:]))  # mean of last 10 scores
            print(f"{label} trace {i}: mean(last10)={all_scores[-1]:.6f}, max={np.max(scores):.6f}, min={np.min(scores):.6f}")
        if all_scores:
            print(f"{label} summary: mean={np.mean(all_scores):.6f}, median={np.median(all_scores):.6f}")

    print("=== MALICIOUS SAMPLES ===")
    summarize(sample_mal, "MAL")
    print("\n=== BENIGN SAMPLES ===")
    summarize(sample_ben, "BENIGN")

if __name__ == "__main__":
    main()
