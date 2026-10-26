import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def setup_chromadb(persist_directory="./chroma_db"):
    """Initializes a persistent ChromaDB client."""
    client = chromadb.PersistentClient(path=persist_directory)
    
    collection = client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    logging.info(f"✅ ChromaDB initialized. Collection: {collection.name}")
    return client, collection

def ingest_chunks(collection, chunks, embeddings, ids, metadatas=None, batch_size=1000):
    """
    Adds text chunks, embeddings, and OPTIONAL metadata into ChromaDB in batches.
    """
    if not chunks or not embeddings:
        logging.warning("No data provided to ingest.")
        return

    if ids is None:
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
    total_chunks = len(chunks)
    logging.info(f"Starting ingestion of {total_chunks} chunks into ChromaDB (Batch size: {batch_size})...")
    
    for i in range(0, total_chunks, batch_size):
        batch_end = i + batch_size
        batch_metadatas = metadatas[i:batch_end] if metadatas is not None else None
        
        collection.add(
            documents=chunks[i:batch_end],
            embeddings=embeddings[i:batch_end],
            ids=ids[i:batch_end],
            metadatas=batch_metadatas
        )
        
        logging.info(f"   - Ingested batch {i//batch_size + 1} ({len(chunks[i:batch_end])} chunks)")
    
    logging.info(f"✅ Successfully ingested {collection.count()} total documents into the database.")

def search_database(collection, query_embedding, n_results=5, metadata_filter=None):
    """
    Searches the database. Now supports optional metadata filtering (e.g., {"sentiment": "Positive"}).
    """
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    if metadata_filter:
        query_params["where"] = metadata_filter
        
    results = collection.query(**query_params)
    return results