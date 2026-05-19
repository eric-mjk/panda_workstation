import socket

HOST = "0.0.0.0"
PORT = 5050

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening on {HOST}:{PORT} ...")

    conn, addr = s.accept()
    with conn:
        print(f"Connected from {addr}")
        conn.sendall(b"hello from local server\n")
        print("Sent hello — connection OK")
