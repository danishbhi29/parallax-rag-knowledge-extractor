import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def setup_chromadb(persist_directory="./chroma_db"):
    """
    Initializes a persistent ChromaDB client.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get the collection
    collection = client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"} # Cosine similarity is best for text
    )
    
    logging.info(f"✅ ChromaDB initialized. Collection: {collection.name}")
    return client, collection

def ingest_chunks(collection, chunks, embeddings, ids, batch_size=1000):
    """
    Adds the text chunks and their embeddings into the ChromaDB collection IN BATCHES.
    This prevents ChromaDB's max batch size limit errors.
    """
    if not chunks or not embeddings:
        logging.warning("No data provided to ingest.")
        return

    if ids is None:
        ids = [f"chunk_{i}" for i in range(len(chunks))]

    total_chunks = len(chunks)
    logging.info(f"Starting ingestion of {total_chunks} chunks into ChromaDB (Batch size: {batch_size})...")
    
    # Loop through the data in steps of 'batch_size'
    for i in range(0, total_chunks, batch_size):
        batch_end = i + batch_size
        
        # Slice the lists to get the current batch
        batch_chunks = chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        # Add the batch to the database
        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            ids=batch_ids
        )
        
        logging.info(f"   - Ingested batch {i//batch_size + 1} ({len(batch_chunks)} chunks)")
    
    logging.info(f"✅ Successfully ingested {collection.count()} total documents into the database.")

def search_database(collection, query_embedding, n_results=5):
    """
    Searches the database for the most similar chunks to a query.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results