#Handles routes

#Import libraries
from fastapi import FastAPI, UploadFile, File
from typing import List
from app.ingestion import process_pdfs

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Server running! Use /ingest to upload PDFs."}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(..., description="Upload one or more PDF files")):
    result = await process_pdfs(files)
    return result