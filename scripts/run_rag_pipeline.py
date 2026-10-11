import os
import sys
import json
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chunking import create_chunks
from embeddings import generate_embeddings
from vector_db import setup_chromadb, ingest_chunks, search_database
from llm_client import generate_answer
from nlp_analysis import extract_topics_and_sentiments

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_pipeline():
    logging.info("="*70)
    logging.info("🚀 Starting Week 4 RAG Pipeline (with NLP Metadata & Filtering)")
    logging.info("="*70)

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
            full_text = f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
            documents.append(full_text)
            
    logging.info(f"✅ Loaded {len(documents)} documents.")

    # 2. Text Chunking
    logging.info("\n✂️ Step 1: Chunking text...")
    chunks = create_chunks(documents, chunk_size=300, chunk_overlap=50)
    logging.info(f"✅ Generated {len(chunks)} chunks.")

    # 3. NLP Analysis (Week 4 Core Task)
    logging.info("\n🧠 Step 2: Extracting Topics and Sentiments (NLP Analysis)...")
    # Note: Running this on all 26k chunks might take a few minutes. 
    # For speed, we can run it on the parent documents and map to chunks, 
    # but for simplicity and accuracy per chunk, we run it on chunks.
    nlp_metadata = extract_topics_and_sentiments(chunks)
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 4. Generate Embeddings
    logging.info("\n🔢 Step 3: Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    # 5. Ingest into ChromaDB WITH Metadata
    logging.info("\n🗄️ Step 4: Ingesting into ChromaDB with Metadata...")
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
    
    ingest_chunks(collection, chunks, embeddings, ids=chunk_ids, metadatas=nlp_metadata, batch_size=1000)

    # 6. End-to-End RAG Flow & Filtered Retrieval Demo (Week 4 Core Task)
    logging.info("\n🔍 Step 5: Running Filtered Retrieval & End-to-End RAG Flow...")
    
    test_query = "How do neural networks learn and optimize?"
    logging.info(f"❓ User Query: '{test_query}'")
    
    # DEMO: Let's say we ONLY want to search in documents that have a "Positive" sentiment
    # and belong to a topic related to "neural" or "learning" (we'll just filter by Positive sentiment for this demo)
    sentiment_filter = {"sentiment": "Positive"}
    logging.info(f"🎯 Applying Metadata Filter: {sentiment_filter}")
    
    e2e_start = time.time()

    # A. Filtered Retrieval Phase
    search_start = time.time()
    query_embedding = generate_embeddings([test_query])[0]
    
    # Notice we pass metadata_filter here!
    search_results = search_database(
        collection, 
        query_embedding=query_embedding, 
        n_results=3, 
        metadata_filter=sentiment_filter
    )
    search_end = time.time()
    retrieval_latency = (search_end - search_start) * 1000
    
    context = " ".join(search_results['documents'][0])
    
    # B. Generation Phase (LLM)
    logging.info("\n🧠 Step 6: Generating Answer via LLM...")
    llm_result = generate_answer(context, test_query)
    generation_latency = llm_result["latency_ms"]

    e2e_end = time.time()
    total_latency = (e2e_end - e2e_start) * 1000

    # 7. Final Results & Logging
    logging.info("\n" + "="*70)
    logging.info("📊 WEEK 4 LATENCY & FILTERED RESULTS REPORT")
    logging.info("="*70)
    logging.info(f"⏱️ Retrieval Latency (ChromaDB + Filter): {retrieval_latency:.2f} ms")
    logging.info(f"⏱️ Generation Latency (LLM):             {generation_latency:.2f} ms")
    logging.info(f"⏱️ TOTAL End-to-End Latency:             {total_latency:.2f} ms")
    logging.info("-" * 70)
    
    if llm_result["error"]:
        logging.error(f"❌ LLM Error: {llm_result['error']}")
    else:
        logging.info(f"✅ AI Answer:\n{llm_result['answer']}")
        
    logging.info("="*70)
    logging.info("🎉 Week 4 Pipeline Execution Successful!")
    logging.info("="*70)

if __name__ == "__main__":
    run_pipeline()