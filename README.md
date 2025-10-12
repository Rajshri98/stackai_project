# StackAI RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) system built with FastAPI, featuring advanced document processing, semantic search, and intelligent question answering capabilities.

## Features

- **PDF Document Processing**: Extract and process text from PDF files using PyMuPDF
- **Advanced RAG Pipeline**: Multi-stage retrieval with semantic and keyword-based search
- **LLM Integration**: Mistral AI for intelligent answer generation
- **Modern UI**: Beautiful, responsive interface with dark/light mode
- **Real-time Chat**: Interactive document querying with citations
- **Smart Chunking**: Character-based text chunking with overlap for context preservation

## Architecture

### 1. Data Ingestion

This step sets up the backend that handles uploading and processing PDF files.  
It's the foundation of the Retrieval-Augmented Generation (RAG) system, preparing clean, searchable text data for later query and generation stages.

#### What It Does
- Upload one or more PDF files through the `/ingest` endpoint
- Extract text from each file using **PyMuPDF**
- Split long text into overlapping chunks to preserve context
- Store all chunks locally in `data/chunks_metadata.json`
- Generate and store vector embeddings in `data/vector_index.npy`
- Return a JSON response with the total number of chunks created

**Example Response:**
```json
{"status": "ok", "chunks_created": 5, "files_processed": 2, "embedding_dimension": 300}
```

### 2. Query Processing

This step determines whether a user's question requires searching the uploaded PDFs and reformats it to improve retrieval accuracy.  
It ensures only meaningful queries trigger document retrieval, while also improving how the query matches relevant text chunks.

#### What It Does
- Detects whether the user's query should trigger a knowledge base search
  - Example:
    - "hello" or "thanks" → no search triggered
    - "What is overfitting?" → search triggered
- Transforms natural language queries into retrieval-friendly formats for better matching
  - Example:
    - "What is regression?" → "regression definition explanation"
    - "How to train a model?" → "steps process for training a model"
- Passes the processed query to the retrieval module for further search and generation

#### Endpoint
`POST /query`

**Example Request:**
```json
{"query": "What is generalization in machine learning?"}
```

**Example Response:**
```json
{
  "intent": "kb",
  "answer": "Generalization refers to how well a model performs on new, unseen data...",
  "citations": [{"chunk": 25, "score": 0.12}]
}
```

### 3. Semantic Search

This step designs the retrieval mechanism that searches the ingested PDF chunks using the processed query.  
It combines **semantic** and **keyword-based** matching to ensure accurate, context-aware retrieval of relevant text.

#### What It Does

- Performs a **hybrid search** that blends:
  - **Semantic similarity (TF-IDF cosine)** - captures meaning and context
  - **Keyword overlap (Jaccard index)** - ensures exact term matching
- Each document chunk receives:
  - `semantic_score` → contextual similarity
  - `keyword_score` → keyword overlap
  - `combined_score` → weighted score (0.6 × semantic + 0.4 × keyword)
- Chunks are ranked by combined score, and only those above a confidence threshold are kept
- If no chunk passes the threshold, the system returns **"insufficient evidence"**

### 4. Post-Processing

This step refines the retrieved chunks by **merging**, **re-ranking**, and **filtering** results to ensure only the most relevant, diverse, and evidence-backed context is used for answer generation.  
It improves retrieval performance and reduces redundancy before passing data to the language model.

#### What It Does

- **Merges and re-ranks results** using:
  - **Reciprocal Rank Fusion (RRF):** balances both semantic and keyword rankings
  - **Maximum Marginal Relevance (MMR):** ensures diversity by penalizing redundant chunks
- **Stitches adjacent chunks** for smoother context continuity
- **Applies evidence thresholding** - if top results don't meet a confidence score, the system returns  
  `"insufficient evidence"` instead of hallucinating an answer
- **Preserves detailed retrieval metrics** in the final output:
  - `semantic_score` → contextual similarity
  - `keyword_score` → keyword overlap
  - `combined_score` → hybrid weighted score
  - `rrf` → rank fusion score

### 5. Generation

This step uses the **Mistral AI** language model to generate grounded, evidence-based answers from the retrieved context.

#### What It Does

- Builds a structured prompt combining the user's question and the top-ranked context chunks
- Calls the Mistral API (`mistral-small-latest`) to generate concise answers strictly based on that context
- Enforces safety through:
  - **Evidence check:** Returns "insufficient evidence" if context lacks support
  - **Hallucination filter:** Flags unsupported claims (e.g., random URLs or figures)
- Preserves citation details (`CHUNK IDs`) for full transparency

#### Example Output
```json
{
  "intent": "kb",
  "answer": "Overfitting occurs when...",
  "citations": [
    {"chunk": 23, "semantic_score": 0.135, "keyword_score": 0.025, "combined_score": 0.091, "rrf": 0.030}
  ]
}
```

## Setup

### Prerequisites

- Python 3.8+
- Mistral AI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stackai_project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env
```

### Mistral API Setup

This step uses **Mistral AI** for generating answers from the retrieved context.  
You need to create your own **Mistral API key** for this project.

Before running the server, set your API key in the terminal:

```bash
export MISTRAL_API_KEY=your_key_here
```

#### Verify the Key

Check if your Mistral API key is set correctly:

```bash
echo $MISTRAL_API_KEY
```

### Usage

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the **Swagger UI** at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Access the **Web Interface** at:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Test the **`/query`** endpoint by entering a question after uploading PDFs.

## API Endpoints

- `GET /` - Web interface
- `GET /api` - API status
- `POST /ingest` - Upload PDF files
- `POST /query` - Query documents

## Technology Stack

- **Backend**: FastAPI, Python
- **PDF Processing**: PyMuPDF (fitz)
- **Embeddings**: Custom TF-IDF + Word Embeddings
- **LLM**: Mistral AI API
- **Frontend**: HTML, CSS, JavaScript
- **Storage**: JSON + NumPy arrays

## Project Structure

```
stackai_project/
├── app/                    # Backend modules
│   ├── main.py            # FastAPI application
│   ├── ingestion.py       # PDF processing
│   ├── query.py          # RAG pipeline
│   ├── embeddings.py     # Custom embeddings
│   ├── reranker.py       # LLM reranking
│   ├── store.py          # Data persistence
│   └── utils_text.py     # Text utilities
├── Frontend/static/       # UI components
│   ├── index.html        # Main interface
│   ├── script.js         # JavaScript
│   └── style.css         # Styling
├── data/                  # Generated data
├── constants.py          # Configuration
├── settings.py           # Environment variables
└── requirements.txt      # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.