import socket
import threading
import os

class Peer:
    def __init__(self, name, address, port, tracker_address, tracker_port):
        self.name = name
        self.address = address
        self.port = port
        self.files = {}
        self.server_socket = None
        self.tracker_address = tracker_address
        self.tracker_port = tracker_port
        self.peers = []
        
        # Register with the tracker
        self.register_with_tracker()

        # Discover other peers from the tracker
        self.discover_peers_from_tracker()

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.address, self.port))
        self.server_socket.listen(5)

        print(f"Peer {self.address}:{self.port} is running as a server...")

        while True:
            client_socket, _ = self.server_socket.accept()
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            
    def register_with_tracker(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tracker_socket:
            tracker_socket.connect((self.tracker_address, self.tracker_port))
            message = f"REGISTER:{self.address}:{self.port}:{self.name}"
            tracker_socket.send(message.encode())

    def discover_peers_from_tracker(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tracker_socket:
            tracker_socket.connect((self.tracker_address, self.tracker_port))
            print('Connected to tracker!')
            tracker_socket.send("LIST".encode())
            response = tracker_socket.recv(1024).decode()
            parts = response.split(",")
            self.peers = [(address, int(port), name) for address, port, name in [p.split(":") for p in parts]]

    def handle_client(self, client_socket):
        data = client_socket.recv(1024).decode()
        parts = data.split(":")
        action = parts[0]

        if action == "SEARCH":
            keyword = parts[1]
            results = self.search_in_self(keyword)
            response = ",".join([f"{filename}:{size}" for filename, size in results])
            response += f" from ({self.address}:{self.port})"
            client_socket.send(response.encode())

        elif action == "SHARE":
            filename = parts[1]
            if filename in self.files:
                file_size = self.files[filename]
                response = f"{filename}:{file_size}"
                client_socket.send(response.encode())

                with open(filename, "rb") as file:
                    while True:
                        data = file.read(1024)
                        if not data:
                            break
                        client_socket.send(data)
            
        client_socket.shutdown(socket.SHUT_RDWR)
        client_socket.close()

    
    def search_in_self(self, keyword):
        results = []
        for filename, size in self.files.items():
            if keyword in filename:
                results.append((filename, size))
        return results


    def search_files(self, keyword):
        self.discover_peers_from_tracker()  # Get the list of peers from the tracker
        results = []
        for peer_address, peer_port, peer_name in self.peers:
            # print("Requesting to:", peer_address, peer_port, peer_name)
            if (peer_address, peer_port) != (self.address, self.port):  # Exclude self
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    try:
                        client_socket.settimeout(5)  # Set a timeout for the connection attempt
                        client_socket.connect((peer_address, peer_port))
                        # print("Connected to:", peer_address, peer_port, peer_name)
                        client_socket.send(f"SEARCH:{keyword}".encode())
                        response = client_socket.recv(1024).decode()
                        print("response",response)

                        parts = response.split(" ")
                        ip_port = parts[2][1:-1]
                        for files in parts[0].split(","):
                            filename, size = files.split(":")
                            results.append((filename, int(size),ip_port))
                    
                    except Exception as e:
                        print(f"An error occurred while searching: {e}")
                    finally:
                        client_socket.close()  # Close the socket after the search request
        return results

    def add_file(self, filename, size):
        self.files[filename] = size

    def share_file(self, filename):
        if filename in self.files:
            return self.files[filename]
        return None

    def download_file(self, filename, ip_port):
        # peer_address = input("Enter the IP address of the peer with the file: ")
        # peer_port = int(input("Enter the port number of the peer: "))
        print(ip_port, type(ip_port))
        peer_address = ip_port.split(":")[0]
        peer_port = int(ip_port.split(":")[1])
        print(f"In Download, peer ip:{peer_address}, port: {peer_port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                client_socket.connect((peer_address, peer_port))
                client_socket.send(f"SHARE:{filename}".encode())
                response = client_socket.recv(1024).decode()

                parts = response.split(":")
                if len(parts) == 2:
                    downloaded_filename = parts[0].split('/')[-1]
                    file_size = int(parts[1])

                    with open(downloaded_filename, "wb") as file:
                        received_bytes = 0
                        print("donwloadin...",end="")
                        while received_bytes < file_size:
                            print("...",end="")
                            data = client_socket.recv(1024)
                            if not data:
                                break
                            file.write(data)
                            received_bytes += len(data)

                    print(f"\nFile '{downloaded_filename}' downloaded successfully.")
                else:
                    print("File not found on the peer.")
            except Exception as e:
                print(f"An error occurred: {e}")

def user_interface(peer):
    while True:
        print("\n1. Share a file")
        print("2. Search for a file")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == "1":
            filename = input("Enter the name of the file to share: ")
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                peer.add_file(filename, file_size)
                print(f"{filename} has been shared.")
            else:
                print("File not found.")

        elif choice == "2":
            keyword = input("Enter a keyword to search for files: ")
            results = peer.search_files(keyword)
            print("results: ",results)
            if results:
                print("Search Results:")
                for i, (filename, size, ip_port) in enumerate(results, start=1):
                    print(f"{i}. {filename} ({size} bytes)")
                choice = input("Enter the number of the file to download (or press Enter to go back): ")
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(results):
                        filename, _, ip_port = results[index]
                        peer.download_file(filename,ip_port)
                    else:
                        print("Invalid selection.")
            else:
                print("No matching files found in the network.")

        elif choice == "3":
            break

if __name__ == "__main__":
    name = "p1 api test"
    address = "0.0.0.0"
    port = 9988
    tracker_address = "0.0.0.0"
    tracker_port = 9999

    peer = Peer(name, address, port, tracker_address, tracker_port)

    # Register with the tracker
    peer.register_with_tracker()

    # Discover other peers from the tracker
    peer.discover_peers_from_tracker()

    threading.Thread(target=peer.start_server).start()
    user_interface(peer)
