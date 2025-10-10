#Handles routes

#Import libraries
from fastapi import FastAPI, UploadFile, File
from typing import List
from app.ingestion import process_pdfs
from app.query import answer_query
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Server running! Use /ingest to upload PDFs."}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(..., description="Upload one or more PDF files")):
    result = await process_pdfs(files)
    return result

class QueryInput(BaseModel):
    query: str

@app.post("/query")
def query(payload: QueryInput):
    return answer_query(payload.query)