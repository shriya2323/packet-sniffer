from scapy.all import sniff

def packet_callback(pkt):
    print(pkt.summary())

# Capture 5 packets as a test
sniff(count=5, prn=packet_callback)
