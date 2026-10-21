import sys
import os

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Tests if the /health endpoint returns a healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "documents_in_db" in response.json()

def test_query_success():
    """Tests a valid query to the /query endpoint."""
    payload = {
        "query": "What is machine learning?",
        "sentiment_filter": "Positive",
        "n_results": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "metadata" in data
    assert "total_latency_ms" in data["metadata"]

def test_query_validation_error_422():
    """Tests that a query shorter than 3 characters returns a 422 Unprocessable Entity error."""
    payload = {
        "query": "AI", # Too short (min_length=3 in Pydantic model)
        "n_results": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422

def test_query_bad_request_400():
    """Tests that an invalid sentiment filter returns a 400 Bad Request error."""
    payload = {
        "query": "What is machine learning?",
        "sentiment_filter": "SuperHappy", # Invalid sentiment
        "n_results": 2
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 400
    assert "Invalid sentiment_filter" in response.json()["detail"]