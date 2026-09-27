import sys
import os
import chromadb
import uuid

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vector_db import setup_chromadb, ingest_chunks, search_database

def test_ingest_and_count():
    """Tests if data is successfully added to the database."""
    # Create a fresh in-memory client and a UNIQUE collection name
    client = chromadb.Client()
    collection_name = f"test_count_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(name=collection_name)
    
    chunks = ["AI is great.", "Data science is fun."]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] # Dummy 3D vectors
    
    ingest_chunks(collection, chunks, embeddings, ids=["id1", "id2"])
    
    # Check if the database actually holds exactly 2 items
    assert collection.count() == 2

def test_ingest_with_batching():
    """Tests that the batching logic works correctly for larger datasets."""
    # Create a fresh in-memory client and a UNIQUE collection name
    client = chromadb.Client()
    collection_name = f"test_batch_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Create 5 dummy chunks and embeddings
    chunks = [f"Chunk {i}" for i in range(5)]
    embeddings = [[0.1, 0.2, 0.3] for _ in range(5)]
    ids = [f"id_{i}" for i in range(5)]
    
    # Force batch_size to 2 to test the looping logic
    ingest_chunks(collection, chunks, embeddings, ids=ids, batch_size=2)
    
    # Check if all 5 items were ingested successfully
    assert collection.count() == 5

def test_semantic_search():
    """Tests if the database can actually find similar items."""
    # Create a fresh in-memory client and a UNIQUE collection name
    client = chromadb.Client()
    collection_name = f"test_search_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(name=collection_name)
    
    chunks = ["The cat sat on the mat.", "The stock market crashed."]
    # Dummy embeddings
    embeddings = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]] 
    
    ingest_chunks(collection, chunks, embeddings, ids=["cat", "stock"])
    
    # Search for something similar to the first chunk
    results = search_database(collection, query_embedding=[0.9, 0.1, 0.0], n_results=1)
    
    # The top result should be the cat chunk
    assert "cat" in results['ids'][0]
    assert "The cat sat on the mat." in results['documents'][0]