# client/stream_client.py
import requests, time, random, socket
SERVER = "http://127.0.0.1:8000/infer"
HOST = socket.gethostname()

EVENTS = ['read_file','write_file','open_socket','close_socket','create_process','terminate_process','load_library','delete_file']

def simulate(session_id=None, n=200, delay=0.1):
    sid = session_id or HOST
    for _ in range(n):
        ev = random.choice(EVENTS)
        payload = {"session_id": sid, "event_name": ev, "ts": time.time()}
        try:
            r = requests.post(SERVER, json=payload, timeout=2)
            print(r.json())
        except Exception as e:
            print("Error:", e)
        time.sleep(delay)

if __name__ == "__main__":
    simulate(n=100, delay=0.05)
