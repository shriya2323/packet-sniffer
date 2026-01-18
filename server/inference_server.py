# server/inference_server.py
import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from fastapi import FastAPI
from pydantic import BaseModel
from collections import deque
import numpy as np

from envs.feature_ops import SlidingWindow, EVENT_IDX
from agents.model_loader import load_model

app = FastAPI(title="RL Malware Inference Server (demo)")

model = load_model()  # returns callable model (TorchScript or nn.Module)

# in-memory state per session
windows = {}
recent_events = {}
RECENT_MAX = 500

class EventIn(BaseModel):
    session_id: str
    event_name: str
    ts: float = None

def compute_score(win: SlidingWindow) -> float:
    obs = win.get_observation()
    import torch
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    arr = out.cpu().numpy().squeeze().tolist()
    if isinstance(arr, (list, tuple)):
        score = float(arr[1]) if len(arr) > 1 else float(arr[0])
    else:
        score = float(arr)
    return score

@app.post("/infer")
def infer(event: EventIn):
    sid = event.session_id
    if sid not in windows:
        windows[sid] = SlidingWindow()
    win = windows[sid]
    win.push_event(event.event_name)
    score = compute_score(win) if model is not None else 0.0
    decision = 1 if score >= 0.5 else 0

    recent_events.setdefault(sid, deque(maxlen=RECENT_MAX)).append({
        "ts": event.ts or time.time(),
        "event": event.event_name,
        "score": score,
        "decision": decision
    })
    return {"session_id": sid, "decision": decision, "malicious_score": score, "timestamp": time.time()}

@app.get("/recent/{session_id}")
def recent(session_id: str, limit: int = 50):
    return list(recent_events.get(session_id, deque(maxlen=RECENT_MAX)))[-limit:]

@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    if session_id in windows:
        windows[session_id].reset()
    recent_events.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}

@app.get("/explain/{session_id}")
def explain(session_id: str, top_k: int = 3):
    if session_id not in windows:
        return {"error": "session not found"}
    win = windows[session_id]
    base = compute_score(win)
    original = [w.copy() for w in win.window]
    deltas = []
    for i in range(len(original)):
        saved = win.window[i].copy()
        win.window[i] = np.zeros_like(saved)
        score = compute_score(win)
        delta = base - score
        deltas.append((i, float(delta), saved.tolist()))
        win.window[i] = saved
    win.window = original
    # map one-hot to event name
    idx2event = {v:k for k,v in EVENT_IDX.items()}
    deltas.sort(key=lambda x: x[1], reverse=True)
    top = []
    for idx, impact, onehot in deltas[:top_k]:
        ev = None
        for j,v in enumerate(onehot):
            if v == 1.0:
                ev = idx2event.get(j, f"evt_{j}")
                break
        top.append({"index": idx, "impact": impact, "event_name": ev})
    return {"session_id": session_id, "base_score": base, "top_contributors": top}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.inference_server:app", host="0.0.0.0", port=8000)
