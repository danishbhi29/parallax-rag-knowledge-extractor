import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from embeddings import generate_embeddings
from vector_db import setup_chromadb, search_database

logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_precision_recall_at_k(retrieved_docs, relevant_doc_indices, k=3):
    """Calculates Precision@K and Recall@K for a single query."""
    retrieved_at_k = retrieved_docs[:k]
    
    # Precision@K: How many of the top K retrieved docs are actually relevant?
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc['index'] in relevant_doc_indices)
    precision_at_k = relevant_retrieved / k
    
    # Recall@K: How many of the total relevant docs did we manage to retrieve in top K?
    total_relevant = len(relevant_doc_indices)
    recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    return precision_at_k, recall_at_k

def run_evaluation():
    logging.info("="*60)
    logging.info("📊 Starting Retrieval Evaluation (Precision@K & Recall@K)")
    logging.info("="*60)
    
    client, collection = setup_chromadb(persist_directory="./chroma_db")
    
    # Mock Test Set: Queries and the indices of documents that SHOULD be retrieved
    # (In a real scenario, this comes from a manually labeled dataset)
    test_set = [
        {
            "query": "What is the role of attention mechanisms in transformers?",
            "relevant_indices": [0, 1, 2] # Mock indices for demonstration
        },
        {
            "query": "How do neural networks learn and optimize?",
            "relevant_indices": [0, 1] 
        }
    ]
    
    k = 3
    total_precision = 0.0
    total_recall = 0.0
    
    for i, test_case in enumerate(test_set):
        logging.info(f"\n🔍 Evaluating Query {i+1}: '{test_case['query']}'")
        
        # 1. Embed query
        query_embedding = generate_embeddings([test_case['query']])[0]
        
        # 2. Retrieve top K documents
        results = search_database(collection, query_embedding=query_embedding, n_results=k)
        retrieved_docs = results['documents'][0]
        
        # For evaluation, we simulate checking if retrieved docs match our ground truth
        # Here we mock the 'index' based on retrieval order for demonstration
        retrieved_with_indices = [{"text": doc, "index": idx} for idx, doc in enumerate(retrieved_docs)]
        
        # 3. Calculate Metrics
        precision, recall = calculate_precision_recall_at_k(
            retrieved_with_indices, 
            test_case['relevant_indices'], 
            k=k
        )
        
        total_precision += precision
        total_recall += recall
        
        logging.info(f"   - Precision@{k}: {precision:.2f}")
        logging.info(f"   - Recall@{k}:    {recall:.2f}")
        
    avg_precision = total_precision / len(test_set)
    avg_recall = total_recall / len(test_set)
    
    logging.info("\n" + "="*60)
    logging.info("📈 EVALUATION SUMMARY")
    logging.info("="*60)
    logging.info(f"Average Precision@{k}: {avg_precision:.2f}")
    logging.info(f"Average Recall@{k}:    {avg_recall:.2f}")
    logging.info("="*60)
    logging.info("✅ Retrieval Evaluation Complete!")

if __name__ == "__main__":
    run_evaluation()