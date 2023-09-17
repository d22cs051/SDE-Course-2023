import socket
import threading
import time

class TrackerServer:
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.peers = set()  # Store registered peers' information (address, port, name)
        self.check_interval = 60  # Check peers every 60 seconds

    def start(self):
        # Start the thread for periodic peer checks
        threading.Thread(target=self.periodic_peer_check).start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.address, self.port))
            server_socket.listen()

            print(f"Tracker server is listening on {self.address}:{self.port}")

            while True:
                client_socket, client_address = server_socket.accept()
                threading.Thread(target=self.handle_client, args=(client_socket, client_address)).start()

    def handle_client(self, client_socket, client_address):
        with client_socket:
            try:
                data = client_socket.recv(1024).decode()
                parts = data.split(":")
                action = parts[0]

                if action == "REGISTER":
                    peer_address = parts[1]
                    peer_port = int(parts[2])
                    peer_name = parts[3]

                    if (peer_address, peer_port, peer_name) in self.peers:
                        return
                    self.peers.add((peer_address, peer_port, peer_name))
                    print(f"Registered peer: {peer_name} ({peer_address}:{peer_port})")

                elif action == "LIST":
                    peer_list = ",".join([f"{address}:{port}:{name}" for address, port, name in self.peers])
                    client_socket.send(peer_list.encode())

            except Exception as e:
                print(f"An error occurred: {e}")

    def periodic_peer_check(self):
        while True:
            time.sleep(self.check_interval)
            self.check_unresponsive_peers()

    def check_unresponsive_peers(self):
        current_time = time.time()
        unresponsive_peers = set()

        for peer_address, peer_port, peer_name in self.peers:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check_socket:
                check_socket.settimeout(1)  # Set a timeout for the connection attempt
                try:
                    check_socket.connect((peer_address, peer_port))
                except (socket.timeout, ConnectionRefusedError):
                    # Peer is unresponsive
                    unresponsive_peers.add((peer_address, peer_port, peer_name))

        while self.peers:
            # Remove unresponsive peers from the set
            self.peers.difference_update(unresponsive_peers)
            for peer_address, peer_port, peer_name in unresponsive_peers:
                print(f"Removed unresponsive peer: {peer_name} ({peer_address}:{peer_port})")

if __name__ == "__main__":
    tracker_address = "0.0.0.0"  # Replace with your desired address
    tracker_port = 9999  # Replace with your desired port

    tracker_server = TrackerServer(tracker_address, tracker_port)
    tracker_server.start()
