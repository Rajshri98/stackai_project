## **1. Data Ingestion**

This step sets up the backend that handles uploading and processing PDF files.  
It’s the foundation of the Retrieval-Augmented Generation (RAG) system — preparing clean, searchable text data for later query and generation stages.

---

### **What It Does**
- Upload one or more PDF files through the `/ingest` endpoint.  
- Extract text from each file using **PyPDF2**.  
- Split long text into overlapping chunks to preserve context.  
- Store all chunks locally in `data/index.json`.  
- Return a JSON response with the total number of chunks created.  

**Example Response:**
```json
{"status": "ok", "chunks_created": 5}