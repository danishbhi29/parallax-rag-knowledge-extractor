import sys
import os

# Ensure Python can find the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nlp_analysis import analyze_sentiment, extract_topics_and_sentiments

def test_sentiment_analysis():
    """Tests if VADER correctly identifies positive and negative text."""
    assert analyze_sentiment("This research is incredibly groundbreaking and excellent.") == "Positive"
    assert analyze_sentiment("The results were terrible and failed completely.") == "Negative"
    assert analyze_sentiment("The sky is blue.") == "Neutral"

def test_topic_and_sentiment_extraction():
    """Tests if BERTopic and VADER run together and return correct metadata structure."""
    # BERTopic requires a minimum of ~15 documents to run its internal UMAP algorithm without errors.
    docs = [
        "Machine learning algorithms are improving rapidly and showing great results in image recognition.",
        "Deep learning models failed to converge and the loss increased significantly during training.",
        "Natural language processing is a subfield of linguistics and computer science focusing on text.",
        "Neural networks have revolutionized artificial intelligence and computer vision tasks recently.",
        "The proposed algorithm demonstrates a significant improvement in accuracy over baseline models.",
        "Our experiments show that the new architecture is highly efficient and robust to noise.",
        "The dataset was corrupted, leading to poor performance and unreliable experimental results.",
        "Reinforcement learning agents successfully learned to play complex games with high scores.",
        "Transformer models have become the standard for state-of-the-art natural language understanding.",
        "The optimization process was unstable, causing the model to diverge and produce random outputs.",
        "Computer vision techniques are now widely used in autonomous driving and medical imaging.",
        "The new framework provides a scalable and efficient solution for large-scale data processing.",
        "We observed a drastic decrease in performance when the model was evaluated on out-of-domain data.",
        "Attention mechanisms allow the model to focus on the most relevant parts of the input sequence.",
        "The results of this study are inconclusive and require further investigation and validation."
    ]
    
    metadata = extract_topics_and_sentiments(docs)
    
    # Check if we got metadata for all docs
    assert len(metadata) == len(docs)
    
    # Check if the structure is correct
    assert "topic" in metadata[0]
    assert "sentiment" in metadata[0]
    
    # Check if sentiment was correctly identified in our dummy data
    assert metadata[0]["sentiment"] == "Positive"
    assert metadata[1]["sentiment"] == "Negative"