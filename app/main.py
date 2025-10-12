# FastAPI application: static hosting, PDF ingestion, and query endpoint

# Imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from app.ingestion import process_pdfs
from app.query import answer_query
from pydantic import BaseModel, ValidationError
import os

app = FastAPI()

# Serve static assets for the frontend UI
app.mount("/static", StaticFiles(directory="Frontend/static"), name="static")

@app.get("/")
def home():
    return FileResponse("Frontend/static/index.html")

@app.get("/api")
def api_home():
    return {"message": "Server running! Use /ingest to upload PDFs."}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(..., description="Upload one or more PDF files")):
    result = await process_pdfs(files)
    return result

class QueryInput(BaseModel):
    query: str

@app.post("/query")
def query(payload: QueryInput):
    try:
        if not payload.query or not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        return answer_query(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")