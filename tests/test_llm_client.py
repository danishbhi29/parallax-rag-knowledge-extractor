import sys
import os

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llm_client import generate_answer

def test_successful_answer_generation():
    """Tests that the LLM returns an answer when the context is relevant."""
    context = "Machine learning is a subset of artificial intelligence that focuses on data and algorithms."
    query = "What is machine learning?"
    
    result = generate_answer(context, query)
    
    assert result["error"] is None
    assert result["answer"] is not None
    assert "machine learning" in result["answer"].lower()
    assert result["latency_ms"] > 0

def test_hallucination_defense_out_of_domain():
    """Tests that the LLM refuses to answer if the context is irrelevant (out-of-domain)."""
    context = "The weather today is sunny with a high of 75 degrees."
    query = "How do neural networks backpropagate?"
    
    result = generate_answer(context, query)
    
    assert result["error"] is None
    # The LLM should trigger our strict system prompt rule
    assert "I cannot answer this based on the provided documents." in result["answer"]

def test_hallucination_defense_missing_context():
    """Tests that the LLM refuses to answer if the context is empty."""
    context = ""
    query = "What is the capital of France?"
    
    result = generate_answer(context, query)
    
    assert result["error"] is None
    assert "I cannot answer this based on the provided documents." in result["answer"]