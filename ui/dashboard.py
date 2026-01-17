# ui/dashboard.py
import streamlit as st
import requests, time
from collections import deque

SERVER = "http://127.0.0.1:8000"
st.set_page_config(layout="wide", page_title="RL Live Monitor")

st.title("RL Live Monitor — Interactive")

session_input = st.text_input("Session ID", "host-1")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Reset session"):
        requests.post(f"{SERVER}/reset/{session_input}")
with col2:
    if st.button("Get explanation"):
        try:
            r = requests.get(f"{SERVER}/explain/{session_input}")
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
with col3:
    if st.button("Send test event"):
        payload = {"session_id": session_input, "event_name": "create_process", "ts": time.time()}
        try:
            r = requests.post(f"{SERVER}/infer", json=payload)
            st.write(r.json())
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.header("Recent events")
try:
    r = requests.get(f"{SERVER}/recent/{session_input}")
    events = r.json()
except Exception as e:
    events = []
    st.error("Cannot fetch recent events: " + str(e))

if events:
    for ev in reversed(events[-100:]):
        ts = time.strftime('%X', time.localtime(ev['ts']))
        st.write(f"{ts} — {ev['event']} — score={ev['score']:.6f} — decision={'MAL' if ev['decision']==1 else 'OK'}")
else:
    st.write("No events yet for this session (send via sniffer or simulator).")

st.sidebar.header("Simulator")
if st.sidebar.button("Simulate 50 events"):
    import client.stream_client as sim
    sim.simulate(session_input, n=50, delay=0.05)
    st.experimental_rerun()
