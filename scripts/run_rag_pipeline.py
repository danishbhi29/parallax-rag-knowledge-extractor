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
from llm_client import generate_answer

# Configure logging for clean, professional console output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_pipeline():
    logging.info("="*60)
    logging.info("🚀 Starting Week 3 RAG Pipeline (with LLM Integration)")
    logging.info("="*60)

    # 1. Load Cleaned Data
    processed_path = "data/processed/clean_corpus.jsonl"
    if not os.path.exists(processed_path):
        logging.error("❌ Cleaned data not found. Please run Week 1 process_data.py first.")
        return

    logging.info(f" Loading data from {processed_path}...")
    documents = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
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
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 4. Ingest into ChromaDB
    logging.info("\n🗄️ Step 3: Ingesting into ChromaDB...")
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name="rag_knowledge_base")
    except ValueError:
        pass
    
    collection = client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch ingestion
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=chunk_ids[i:i+batch_size]
        )
    logging.info(f"✅ Successfully ingested {collection.count()} documents.")

    # 5. End-to-End RAG Flow & Latency Benchmarking (WEEK 3 CORE TASK)
    logging.info("\n Step 4: Running End-to-End RAG Flow...")
    
    test_query = "How do neural networks learn and optimize?"
    logging.info(f"❓ User Query: '{test_query}'")
    
    # Start TOTAL End-to-End Timer
    e2e_start = time.time()

    # A. Retrieval Phase
    search_start = time.time()
    query_embedding = generate_embeddings([test_query])[0]
    search_results = search_database(collection, query_embedding=query_embedding, n_results=3)
    search_end = time.time()
    retrieval_latency = (search_end - search_start) * 1000
    
    # Combine retrieved chunks into a single context string
    context = " ".join(search_results['documents'][0])

    # B. Generation Phase (LLM)
    logging.info("\n🧠 Step 5: Generating Answer via LLM...")
    llm_result = generate_answer(context, test_query)
    generation_latency = llm_result["latency_ms"]

    # Stop TOTAL End-to-End Timer
    e2e_end = time.time()
    total_latency = (e2e_end - e2e_start) * 1000

    # 6. Final Results & Logging
    logging.info("\n" + "="*60)
    logging.info("📊 WEEK 3 LATENCY & RESULTS REPORT")
    logging.info("="*60)
    logging.info(f"⏱️ Retrieval Latency (ChromaDB): {retrieval_latency:.2f} ms")
    logging.info(f"⏱️ Generation Latency (LLM):    {generation_latency:.2f} ms")
    logging.info(f"⏱️ TOTAL End-to-End Latency:    {total_latency:.2f} ms")
    logging.info("-" * 60)
    
    if llm_result["error"]:
        logging.error(f"❌ LLM Error: {llm_result['error']}")
    else:
        logging.info(f"✅ AI Answer:\n{llm_result['answer']}")
        
    logging.info("="*60)
    logging.info(" Week 3 Pipeline Execution Successful!")
    logging.info("="*60)

if __name__ == "__main__":
    run_pipeline()