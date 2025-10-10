import json
import os

CHUNK_FILE = "data/index.json"

def save_chunks(chunks):
    os.makedirs("data", exist_ok=True)
    with open(CHUNK_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

def load_chunks():
    if not os.path.exists(CHUNK_FILE):
        return []
    with open(CHUNK_FILE, "r") as f:
        return json.load(f)