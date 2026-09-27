import time
import logging
from sentence_transformers import SentenceTransformer

# Configure logging to print performance metrics to the console
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load the model ONCE globally so we don't reload it every time we call the function
# This is a crucial performance optimization!
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """
    Converts a list of text chunks into numerical vectors (embeddings).
    Logs the time taken to generate them.
    
    :param chunks: List of strings (text chunks).
    :return: List of numerical vectors (embeddings).
    """
    if not chunks:
        logging.warning("No chunks provided for embedding generation.")
        return []

    logging.info(f"Starting embedding generation for {len(chunks)} chunks...")
    
    # Start the timer
    start_time = time.time()
    
    # Generate embeddings using the pre-loaded model
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    # Stop the timer
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Log the performance metrics (Required by Week 2 rules!)
    logging.info(f"✅ Embedding generation complete!")
    logging.info(f"   - Time taken: {time_taken:.2f} seconds")
    logging.info(f"   - Average time per chunk: {(time_taken / len(chunks)) * 1000:.2f} ms")
    
    return embeddings.tolist() # Convert numpy array to standard Python list