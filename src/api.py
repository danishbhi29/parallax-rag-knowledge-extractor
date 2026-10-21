import os
import sys
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from embeddings import generate_embeddings
from vector_db import setup_chromadb, search_database
from llm_client import generate_answer

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Parallax RAG API",
    description="Advanced RAG API with NLP Metadata Filtering and Evaluation",
    version="1.0.0"
)

# Initialize DB at module level (Fixes TestClient startup issues & removes deprecation warnings)
try:
    logger.info("Initializing ChromaDB connection...")
    db_client, collection = setup_chromadb(persist_directory="./chroma_db")
    logger.info(f"✅ ChromaDB loaded successfully. Total documents: {collection.count()}")
except Exception as e:
    logger.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
    db_client = None
    collection = None

# Pydantic model for request validation (Automatically handles 422 Unprocessable Entity errors)
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The search query")
    sentiment_filter: str | None = Field(None, description="Optional: 'Positive', 'Negative', or 'Neutral'")
    n_results: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")

# Custom Exception Handler for 500 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"500 Internal Server Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred."}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and DB are running."""
    if collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return {"status": "healthy", "documents_in_db": collection.count()}

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Main RAG endpoint. Retrieves context and generates an answer.
    Handles 400 (Bad Request) if query is invalid, and 500 for server errors.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or just whitespace.")

    if collection is None:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    start_time = time.time()
    
    try:
        # 1. Generate Query Embedding
        query_embedding = generate_embeddings([request.query])[0]
        
        # 2. Build Metadata Filter (if provided)
        metadata_filter = None
        if request.sentiment_filter:
            valid_sentiments = ["Positive", "Negative", "Neutral"]
            if request.sentiment_filter not in valid_sentiments:
                raise HTTPException(status_code=400, detail=f"Invalid sentiment_filter. Must be one of: {valid_sentiments}")
            metadata_filter = {"sentiment": request.sentiment_filter}
            
        # 3. Retrieve Context
        search_results = search_database(
            collection, 
            query_embedding=query_embedding, 
            n_results=request.n_results, 
            metadata_filter=metadata_filter
        )
        
        if not search_results['documents'] or not search_results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant documents found matching the criteria.")
            
        context = " ".join(search_results['documents'][0])
        
        # 4. Generate Answer via LLM
        llm_result = generate_answer(context, request.query)
        
        if llm_result["error"]:
            raise HTTPException(status_code=500, detail=f"LLM Generation Failed: {llm_result['error']}")
            
        total_latency = (time.time() - start_time) * 1000
        
        # 5. Structured Response
        return {
            "query": request.query,
            "answer": llm_result["answer"],
            "metadata": {
                "retrieval_latency_ms": round((total_latency - llm_result["latency_ms"]), 2),
                "generation_latency_ms": round(llm_result["latency_ms"], 2),
                "total_latency_ms": round(total_latency, 2),
                "applied_filters": metadata_filter
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions (400, 404, etc.)
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during query processing.")