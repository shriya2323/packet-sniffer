#!/usr/bin/env python3
"""
Main entry point for the Packet Sniffer RL project.
This script starts the complete system: server, dashboard, and sniffer in a single command.
"""

import sys
import os
import subprocess
import threading
import time
import signal
import argparse
from pathlib import Path


def start_server(port=2000):
    """Start the inference server in a subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.inference_server:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    print(f"Starting inference server on port {port}...")
    server_process = subprocess.Popen(cmd)
    return server_process


def start_dashboard():
    """Start the Streamlit dashboard in a subprocess."""
    cmd = [sys.executable, "-m", "streamlit", "run", "integrated_sniffer_rl_dashboard.py", "--server.port", "8501"]
    
    print("Starting dashboard on port 8501...")
    dashboard_process = subprocess.Popen(cmd)
    return dashboard_process


def start_sniffer(interface=None, server_port=2000):
    """Start the packet sniffer in a subprocess with sudo privileges."""
    if os.name != 'posix':
        print("Warning: This script is designed for Unix-like systems (Linux/macOS)")
        return None
    
    # Determine the interface automatically if not specified
    if interface is None:
        try:
            # Try to determine the default interface
            route_result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                        capture_output=True, text=True)
            if route_result.returncode == 0:
                # Extract interface from route command
                lines = route_result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split()
                    if 'dev' in parts:
                        idx = parts.index('dev')
                        if idx + 1 < len(parts):
                            interface = parts[idx + 1]
        except:
            print("Could not automatically determine network interface. Please specify one.")
            return None
    
    if interface is None:
        print("No network interface specified and could not determine automatically.")
        print("Please run with --interface option or run the sniffer separately.")
        return None
    
    print(f"Starting packet sniffer on interface: {interface}")
    
    # Create a temporary script that modifies the server URL
    temp_script_content = f'''
import sys
import socket
import requests
import time
from scapy.all import sniff, TCP, Raw, IP

# Set the server URL based on the provided port
SERVER = "http://127.0.0.1:{server_port}/infer"
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
    payload = {{"session_id": HOST_ID, "event_name": event_name, "ts": time.time()}}
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", help="Interface to sniff (optional)", default=None)
    args = parser.parse_args()
    print("Starting sniffer on host:", HOST_ID)
    print("Connecting to server:", SERVER)
    print("Press Ctrl+C to stop.")
    sniff(prn=packet_handler, store=False, iface=args.iface)
'''
    
    # Write the temporary script
    temp_script_path = "/tmp/sniffer_with_port.py"
    with open(temp_script_path, 'w') as f:
        f.write(temp_script_content)
    
    # Use sudo to run the sniffer with elevated privileges
    cmd = [
        'sudo', '-E', 'python3', temp_script_path,
        '--iface', interface
    ]
    
    sniffer_process = subprocess.Popen(cmd)
    return sniffer_process


def main():
    parser = argparse.ArgumentParser(description='Run the complete Packet Sniffer RL system')
    parser.add_argument('--port', type=int, default=2000, help='Port for the inference server (default: 2000)')
    parser.add_argument('--interface', type=str, help='Network interface to sniff (optional, will try to auto-detect)')
    parser.add_argument('--no-server', action='store_true', help='Skip starting the server')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip starting the dashboard')
    parser.add_argument('--no-sniffer', action='store_true', help='Skip starting the packet sniffer')
    
    args = parser.parse_args()
    
    print("🚀 Starting Packet Sniffer RL System")
    print("=" * 50)
    
    processes = []
    
    try:
        # Start server if not disabled
        if not args.no_server:
            server_proc = start_server(args.port)
            processes.append(('server', server_proc))
        
        # Give server a moment to start
        time.sleep(3)
        
        # Start dashboard if not disabled
        if not args.no_dashboard:
            dashboard_proc = start_dashboard()
            processes.append(('dashboard', dashboard_proc))
        
        # Start sniffer if not disabled
        if not args.no_sniffer:
            sniffer_proc = start_sniffer(args.interface, args.port)
            if sniffer_proc:
                processes.append(('sniffer', sniffer_proc))
        
        print("\n✅ All components started!")
        print(f"🌐 Server: http://localhost:{args.port}")
        print("📊 Dashboard: http://localhost:8501")
        print("📡 Sniffer: Listening for packets")
        print("\nPress Ctrl+C to stop all components")
        
        # Wait for all processes to complete (they won't unless interrupted)
        for name, proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all components...")
        
        # Terminate all processes gracefully
        for name, proc in processes:
            print(f"Terminating {name} (PID: {proc.pid})...")
            proc.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Force kill if still running
        for name, proc in processes:
            if proc.poll() is None:  # Still running
                print(f"Force killing {name} (PID: {proc.pid})...")
                proc.kill()
        
        print("👋 All components stopped.")
    except Exception as e:
        print(f"❌ Error running system: {e}")
        # Clean up any running processes
        for name, proc in processes:
            try:
                proc.terminate()
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()