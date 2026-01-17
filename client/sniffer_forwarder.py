# client/sniffer.py
from scapy.all import sniff, TCP, Raw, IP
import socket, requests, time, argparse

SERVER = "http://127.0.0.1:8000/infer"
HOST_ID = socket.gethostname()

def map_packet_to_event(pkt):
    try:
        if TCP in pkt:
            flags = pkt[TCP].flags
            # SYN (open)
            if flags & 0x02:
                return "open_socket"
            # FIN (close)
            if flags & 0x01:
                return "close_socket"
        if Raw in pkt:
            # payload seen -> approximate as read_file/network payload
            return "read_file"
    except Exception:
        pass
    return None

def send_event(event_name):
    payload = {"session_id": HOST_ID, "event_name": event_name, "ts": time.time()}
    try:
        r = requests.post(SERVER, json=payload, timeout=1)
        print("->", r.json())
    except Exception as e:
        print("Send error:", e)

def packet_handler(pkt):
    ev = map_packet_to_event(pkt)
    if ev:
        send_event(ev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", help="Interface to sniff (optional)", default=None)
    args = parser.parse_args()
    print("Starting sniffer on host:", HOST_ID)
    print("Press Ctrl+C to stop.")
    sniff(prn=packet_handler, store=False, iface=args.iface)
