import threading
from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import socket
import os
from starlette.responses import FileResponse
from peer import Peer
import uvicorn


app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "shared"

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SharedFile:
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size


# Create a model for file sharing
class FileShareRequest(BaseModel):
    filename: str
    size: int

# Create a model for search request
class SearchRequest(BaseModel):
    keyword: str

# Create a model for peer information
class PeerInfo(BaseModel):
    name: str
    address: str
    port: int

# Create a peer instance
peer = Peer(name="Peer1", address="0.0.0.0", port=9985, tracker_address="0.0.0.0", tracker_port=9999)

# Start the Peer server in a separate thread
def start_peer_server():
    peer.start_server()

# Start the Peer server in a separate thread
peer_server_thread = threading.Thread(target=start_peer_server)
peer_server_thread.start()

@app.get("/")
async def root():
    return {"message":" peer2peer apis is up and running..."}

@app.post("/share/")
async def share_file(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    shared_files_list = []

    for file in files:
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Check if the file already exists
        # if os.path.exists(file_path):
        #     raise HTTPException(status_code=400, detail=f"File '{filename}' already exists.")

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Add the file to the peer
        file_size = os.path.getsize(file_path)
        peer.add_file(file_path, file_size)
        print("FILE ADDED")
        # Create a SharedFile object to store file info
        shared_file = SharedFile(filename=filename, size=file_size)
        shared_files_list.append(shared_file)

    return {"message": "Files have been shared.", "shared_files": shared_files_list}

@app.post("/search/")
async def search_files(search_request: SearchRequest):
    keyword = search_request.keyword

    # Call the peer's search_files method
    results = peer.search_files(keyword)

    if not results:
        raise HTTPException(status_code=404, detail="No matching files found.")

    return {"results": results}

@app.get("/download/")
async def download_file(filename: str):
    # Check if the file exists in the shared_files
    search_res = [{"filename":filename,"file_size":file_size,"source":file_source_addr} for (filename,file_size,file_source_addr) in peer.search_files(filename)]
    if filename not in [filename.split('/')[-1] for (filename,file_size,file_source_addr) in peer.search_files(keyword=filename)]:
        raise HTTPException(status_code=404, detail="File not found.")

    # Get the file size
    for res in search_res:
        if filename in res["filename"]:
            file_size = res["file_size"]
            break

    # Function to stream file content to the client
    def file_stream():
        with open(filename, "rb") as file:
            while True:
                data = file.read(1024)
                if not data:
                    break
                yield data

    # Create a StreamingResponse to stream the file content to the client
    response = StreamingResponse(file_stream(), media_type="application/octet-stream")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Length"] = str(file_size)

    return response

@app.get("/peers/")
async def get_available_peers():
    # Call the peer's discover_peers_from_tracker method
    peer.discover_peers_from_tracker()

    # print(peer.peers)
    # Return the list of available peers
    available_peers = [{"name": peer[2], "address": peer[0], "port": peer[1]} for peer in peer.peers]
    return {"peers": available_peers}

if __name__ == "__main__":

    # Start the FastAPI application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)