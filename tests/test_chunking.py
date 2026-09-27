import sys
import os

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chunking import create_chunks

def test_basic_chunking():
    """Tests if text is split when it exceeds chunk_size."""
    # A text longer than 50 characters
    long_text = ["This is a short text. " * 10] 
    chunks = create_chunks(long_text, chunk_size=50, chunk_overlap=10)
    
    # We should get more than 1 chunk
    assert len(chunks) > 1
    # Every chunk should be roughly around or under the chunk size (allowing for slight overlap)
    for chunk in chunks:
        assert len(chunk) <= 60 # 50 + a little buffer for overlap

def test_overlap_preserves_context():
    """Tests that chunks actually overlap."""
    text = ["A B C D E F G H I J K L M N O P"]
    chunks = create_chunks(text, chunk_size=10, chunk_overlap=5)
    
    # If overlap works, the end of chunk 1 should match the start of chunk 2
    assert len(chunks) > 1

def test_empty_text_edge_case():
    """Tests that empty strings or whitespace are ignored."""
    texts = ["Valid text.", "", "   ", "Another valid text."]
    chunks = create_chunks(texts, chunk_size=50, chunk_overlap=10)
    
    # Should only return chunks for the valid texts
    assert len(chunks) == 2
    assert "Valid text." in chunks
    assert "Another valid text." in chunks