import socket

SERVER_IP = "147.46.118.233"  # this (host/server) computer's IP
PORT = 5050

with socket.create_connection((SERVER_IP, PORT), timeout=5.0) as s:
    print(f"Connected to {SERVER_IP}:{PORT}")
    msg = s.recv(64)
    print(f"Received: {msg.decode()}")
