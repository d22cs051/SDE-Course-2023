import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt
from peer import Peer
import os

class PeerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.peer = None

    def init_ui(self):
        self.setWindowTitle("Peer P2P File Sharing")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.status_label = QLabel("Status: Not connected to tracker")
        layout.addWidget(self.status_label)

        self.address_label = QLabel("Peer Address:")
        self.address_line_edit = QLineEdit()
        self.address_line_edit.setText("0.0.0.0")
        layout.addWidget(self.address_label)
        layout.addWidget(self.address_line_edit)

        self.port_label = QLabel("Peer Port:")
        self.port_line_edit = QLineEdit()
        self.port_line_edit.setText("9988")
        layout.addWidget(self.port_label)
        layout.addWidget(self.port_line_edit)

        self.tracker_address_label = QLabel("Tracker Address:")
        self.tracker_address_line_edit = QLineEdit()
        self.tracker_address_line_edit.setText("0.0.0.0")
        layout.addWidget(self.tracker_address_label)
        layout.addWidget(self.tracker_address_line_edit)

        self.tracker_port_label = QLabel("Tracker Port:")
        self.tracker_port_line_edit = QLineEdit()
        self.tracker_port_line_edit.setText("9999")
        layout.addWidget(self.tracker_port_label)
        layout.addWidget(self.tracker_port_line_edit)

        connect_button = QPushButton("Connect to Tracker")
        connect_button.clicked.connect(self.connect_to_tracker)
        layout.addWidget(connect_button)
        
        layout.addWidget(QLabel("Shared Files:"))
        self.shared_files_text = QTextEdit()
        layout.addWidget(self.shared_files_text)

        share_file_button = QPushButton("Share File")
        share_file_button.clicked.connect(self.share_file)
        layout.addWidget(share_file_button)

        layout.addWidget(QLabel("Search Keyword:"))
        self.search_keyword_line_edit = QLineEdit()
        layout.addWidget(self.search_keyword_line_edit)

        search_button = QPushButton("Search Files")
        search_button.clicked.connect(self.search_files)
        layout.addWidget(search_button)

        layout.addWidget(QLabel("Search Results:"))
        self.search_results_text = QTextEdit()
        layout.addWidget(self.search_results_text)

        download_button = QPushButton("Download Selected File")
        download_button.clicked.connect(self.download_file)
        layout.addWidget(download_button)

    def start_peer_server(self):
        self.peer.start_server()

    
    def connect_to_tracker(self):
        peer_name = "Peer1"  # You can change this
        peer_address = self.address_line_edit.text()
        peer_port = int(self.port_line_edit.text())
        tracker_address = self.tracker_address_line_edit.text()
        tracker_port = int(self.tracker_port_line_edit.text())

        self.peer = Peer(peer_name, peer_address, peer_port, tracker_address, tracker_port)

        # Start the Peer server in a separate thread
        peer_server_thread = threading.Thread(target=self.start_peer_server)
        peer_server_thread.start()

        # Register with the tracker
        self.peer.register_with_tracker()

        # Discover other peers from the tracker
        self.peer.discover_peers_from_tracker()

        self.status_label.setText("Status: Connected to tracker")

    def share_file(self):
        if self.peer:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Share", "", "All Files (*)", options=options)
            if file_path:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                self.peer.add_file(file_name, file_size)
                self.shared_files_text.append(f"Shared: {file_name} ({file_size} bytes)")
                self.peer.share_file(filename=file_name)

    def search_files(self):
        keyword = self.search_keyword_line_edit.text()
        if self.peer:
            results = self.peer.search_files(keyword)
            self.display_search_results(results)

    def display_search_results(self, results):
        self.search_results_text.clear()
        print("results: ",results)
        for filename, size, source_ip in results:
            self.search_results_text.append(f"{filename} ({size} bytes from {source_ip})")

    def download_file(self):
        selected_text = self.search_results_text.textCursor().selectedText()
        if selected_text:
            print("selected_text: ",selected_text.split())
            filename = selected_text.split()[0]
            source_ip = selected_text.split()[-1][:-1]
            if self.peer:
                self.peer.download_file(filename,source_ip)
                self.status_label.setText(f"Status: Downloading {filename}...")
        self.status_label.setText(f"Status: Downloaded {filename}!!!")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeerGUI()
    window.show()
    sys.exit(app.exec_())
