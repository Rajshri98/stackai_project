#Reads, cleans, chunks, and stores PDF data in data/index.json.

#Extract text from pdfs uploaded
import fitz
import json, os
from typing import List
from fastapi import UploadFile
from app.utils_text import clean_text, chunk_text

async def process_pdfs(files: List[UploadFile]):
    os.makedirs("data", exist_ok=True)
    all_chunks = []

    for file in files:
        pdf = fitz.open(stream=await file.read(), filetype="pdf")
        text = ""

        #extract text from each page
        for page in pdf:
            text += page.get_text("text")
        pdf.close()

        #clean and chunk text
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)

    #save text chunks in local JSON
    with open("data/index.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "chunks_created": len(all_chunks)}