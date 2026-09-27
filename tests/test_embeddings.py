import sys
import os

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from embeddings import generate_embeddings

def test_embedding_dimensions():
    """Tests if the generated embeddings have the correct 384 dimensions."""
    text_chunks = ["This is a test chunk about AI.", "Another test chunk about data."]
    embeddings = generate_embeddings(text_chunks)
    
    # We should have 2 embeddings for 2 chunks
    assert len(embeddings) == 2
    
    # The standard MiniLM model outputs 384 dimensions
    assert len(embeddings[0]) == 384

def test_empty_list_edge_case():
    """Tests that an empty list returns an empty list without crashing."""
    embeddings = generate_embeddings([])
    assert embeddings == []

def test_similar_texts_have_similar_embeddings():
    """Tests that mathematically, similar texts have closer vectors."""
    text_a = ["The cat sat on the mat."]
    text_b = ["A feline rested on the rug."] # Similar meaning
    text_c = ["The stock market crashed today."] # Completely different meaning
    
    emb_a = generate_embeddings(text_a)[0]
    emb_b = generate_embeddings(text_b)[0]
    emb_c = generate_embeddings(text_c)[0]
    
    # Calculate simple Euclidean distance (lower means more similar)
    dist_ab = sum((a - b) ** 2 for a, b in zip(emb_a, emb_b)) ** 0.5
    dist_ac = sum((a - c) ** 2 for a, c in zip(emb_a, emb_c)) ** 0.5
    
    # The distance between similar texts (A and B) should be smaller than different texts (A and C)
    assert dist_ab < dist_ac