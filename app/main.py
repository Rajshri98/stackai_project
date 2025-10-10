#Handles routes

#Import libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from app.ingestion import process_pdfs
from app.query import answer_query
from pydantic import BaseModel, ValidationError

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
    try:
        if not payload.query or not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        return answer_query(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")