from fastapi import FastAPI
import os

worker_id = os.getpid()
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, FastAPI", "worker_number": worker_id}
