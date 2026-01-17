import streamlit as st
import threading
import time
import pandas as pd
from scapy.all import sniff, IP, get_if_list
import random

# ---------------------------
# Global variables
# ---------------------------
captured_packets = []
sniff_thread = None
stop_sniff = threading.Event()

# ---------------------------
# RL Model (Mock Simulation)
# ---------------------------
def rl_based_detection(packet_features):
    """Simulate RL-based malicious detection"""
    threat_score = random.random()
    return "Malicious" if threat_score > 0.8 else "Normal"

# ---------------------------
# Packet Capture Function
# ---------------------------
def packet_handler(packet):
    global captured_packets
    try:
        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            proto = packet[IP].proto
            proto_name = {6: "TCP", 17: "UDP"}.get(proto, "Other")
            size = len(packet)
            result = rl_based_detection({
                "src": src, "dst": dst, "proto": proto_name, "size": size
            })
            captured_packets.append({
                "Time": time.strftime("%H:%M:%S"),
                "Source": src,
                "Destination": dst,
                "Protocol": proto_name,
                "Size": size,
                "Prediction": result
            })
    except Exception as e:
        print(f"Error handling packet: {e}")

def start_sniffing(interface):
    stop_sniff.clear()
    sniff(prn=packet_handler, iface=interface, store=False, stop_filter=lambda x: stop_sniff.is_set())

def stop_sniffing():
    stop_sniff.set()

# ---------------------------
# Streamlit Dashboard
# ---------------------------
st.set_page_config(page_title="RL-based Packet Sniffer", layout="wide")
st.title("🧠 Reinforcement Learning-Based Network Sniffer & Malicious Detection")

st.sidebar.header("⚙️ Controls")

# Detect interfaces using Scapy (more reliable)
interfaces = get_if_list()
selected_interface = st.sidebar.selectbox("Select Network Interface", interfaces)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ Start Capture", use_container_width=True)
stop_btn = col2.button("⏹️ Stop Capture", use_container_width=True)

if "sniff_running" not in st.session_state:
    st.session_state.sniff_running = False

# Start capture
if start_btn and not st.session_state.sniff_running:
    st.session_state.sniff_running = True
    captured_packets.clear()
    stop_sniff.clear()
    sniff_thread = threading.Thread(target=start_sniffing, args=(selected_interface,), daemon=True)
    sniff_thread.start()
    st.sidebar.success(f"✅ Started sniffing on: {selected_interface}")

# Stop capture
if stop_btn and st.session_state.sniff_running:
    stop_sniffing()
    st.session_state.sniff_running = False
    st.sidebar.warning("🛑 Stopped sniffing.")

# Live Packet Display
st.subheader("📡 Live Captured Packets")
placeholder = st.empty()

if st.session_state.sniff_running:
    st.info("🔍 Capturing packets... please wait.")
    time.sleep(2)

if captured_packets:
    df = pd.DataFrame(captured_packets)
    placeholder.dataframe(df.tail(50), use_container_width=True)
else:
    st.info("No packets captured yet. Click 'Start Capture' to begin.")

# Summary statistics
if captured_packets:
    st.subheader("📊 Detection Summary")
    total_packets = len(captured_packets)
    malicious_packets = sum(1 for p in captured_packets if p["Prediction"] == "Malicious")
    st.metric("Total Packets", total_packets)
    st.metric("Malicious Packets", malicious_packets)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip:** Run as Administrator for packet capture on Windows.")
