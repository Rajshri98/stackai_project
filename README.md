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


## **2. Query Processing**

This step determines whether a user’s question requires searching the uploaded PDFs and reformats it to improve retrieval accuracy.  
It ensures only meaningful queries trigger document retrieval, while also improving how the query matches relevant text chunks.

---

### **What It Does**
- Detects whether the user’s query should trigger a knowledge base search.  
  - Example:  
    - “hello” or “thanks” → no search triggered  
    - “What is overfitting?” → search triggered  
- Transforms natural language queries into retrieval-friendly formats for better matching.  
  - Example:  
    - “What is regression?” → “regression definition explanation”  
    - “How to train a model?” → “steps process for training a model”  
- Passes the processed query to the retrieval module for further search and generation.

---

### **Endpoint**
`POST /query`

**Example Request:**
```json
{"query": "What is generalization in machine learning?"}```
```

**Example Response:**
{
  "intent": "kb",
  "answer": "Generalization refers to how well a model performs on new, unseen data...",
  "citations": [{"chunk": 25, "score": 0.12}]
}
```

### **Mistral API Setup**

This step uses **Mistral AI** for generating answers from the retrieved context.  
I created my own **Mistral API key** for this project.

Before running the server, set your API key in the terminal:

```bash
export MISTRAL_API_KEY=your_key_here
```

### **Verify the Key**

Check if your Mistral API key is set correctly:

```bash
echo $MISTRAL_API_KEY

### **Usage**

Start the FastAPI server:

```bash
uvicorn app.main:app --reload

Open the **Swagger UI** at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Test the **`/query`** endpoint by entering a question after uploading PDFs.
```