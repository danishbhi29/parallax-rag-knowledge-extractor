import os
import sys
import json
import time
import logging

# Add src to path to import our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chunking import create_chunks
from embeddings import generate_embeddings
from vector_db import setup_chromadb, ingest_chunks, search_database

# Configure logging for clean, professional console output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_pipeline():
    logging.info("="*60)
    logging.info("🚀 Starting Week 2 RAG Pipeline Execution")
    logging.info("="*60)

    # 1. Load Cleaned Data
    processed_path = "data/processed/clean_corpus.jsonl"
    if not os.path.exists(processed_path):
        logging.error("❌ Cleaned data not found. Please run Week 1 process_data.py first.")
        return

    logging.info(f"📂 Loading data from {processed_path}...")
    documents = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            # Combine title and abstract for better context
            full_text = f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
            documents.append(full_text)
            
    logging.info(f"✅ Loaded {len(documents)} documents.")

    # 2. Text Chunking
    logging.info("\n✂️ Step 1: Chunking text...")
    chunks = create_chunks(documents, chunk_size=300, chunk_overlap=50)
    logging.info(f"✅ Generated {len(chunks)} chunks.")

    # 3. Generate Embeddings
    logging.info("\n🔢 Step 2: Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    
    # Generate unique IDs for each chunk
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 4. Ingest into ChromaDB
    logging.info("\n🗄️ Step 3: Ingesting into ChromaDB...")
    
    # Initialize client
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Clear existing collection safely to prevent duplicates
    try:
        client.delete_collection(name="rag_knowledge_base")
    except ValueError:
        pass  # Collection doesn't exist yet, which is fine
    
    # Create a fresh collection
    collection = client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    logging.info(f"✅ ChromaDB initialized. Collection: {collection.name}")
    
    ingest_chunks(collection, chunks, embeddings, ids=chunk_ids)

    # 5. Semantic Search & Latency Benchmarking (CRITICAL WEEK 2 REQUIREMENT)
    logging.info("\n🔍 Step 4: Benchmarking Semantic Search Latency...")
    
    # A sample query a user might ask
    test_query = "How do neural networks learn and optimize?"
    logging.info(f"   Query: '{test_query}'")
    
    # Embed the query
    query_embedding = generate_embeddings([test_query])[0]
    
    # Measure search latency
    start_search = time.time()
    search_results = search_database(collection, query_embedding=query_embedding, n_results=3)
    end_search = time.time()
    
    latency_ms = (end_search - start_search) * 1000
    
    logging.info(f"✅ Search Complete!")
    logging.info(f"   - Latency: {latency_ms:.2f} milliseconds")
    
    # Safely get the top result snippet
    top_doc = search_results['documents'][0][0] if search_results['documents'] and search_results['documents'][0] else "No results found"
    logging.info(f"   - Top Result Snippet: '{top_doc[:100]}...'")
    
    logging.info("\n" + "="*60)
    logging.info("🎉 Week 2 Pipeline Execution Successful!")
    logging.info("="*60)

if __name__ == "__main__":
    run_pipeline()