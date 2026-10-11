import logging
from bertopic import BERTopic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Initialize VADER Sentiment Analyzer globally for performance
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes sentiment of a text and returns a label (Positive, Negative, Neutral)."""
    if not text or len(text.strip()) < 10:
        return "Neutral" # Edge case: short or empty text
        
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Standard VADER thresholds
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def extract_topics_and_sentiments(documents):
    """
    Uses BERTopic to find themes and VADER for sentiment.
    Returns a list of metadata dictionaries for each document.
    """
    logging.info("🧠 Starting NLP Analysis (Topic Modeling & Sentiment)...")
    
    # 1. Topic Modeling with BERTopic
    logging.info("   - Running BERTopic (this might take 1-2 minutes)...")
    # We use verbose=False to keep console clean, and language="english"
    topic_model = BERTopic(language="english", verbose=False)
    topics, probs = topic_model.fit_transform(documents)
    
    # Get the topic names (e.g., "0_neural_networks_deep_learning")
    topic_info = topic_model.get_topic_info()
    
    # Create a dictionary to map Topic ID to Topic Name
    topic_id_to_name = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    
    # 2. Sentiment Analysis with VADER
    logging.info("   - Running VADER Sentiment Analysis...")
    sentiments = [analyze_sentiment(doc) for doc in documents]
    
    # 3. Combine into a list of metadata dictionaries
    metadata_list = []
    unique_topics = set()
    
    for i in range(len(documents)):
        topic_id = topics[i]
        topic_name = topic_id_to_name.get(topic_id, "Unknown")
        unique_topics.add(topic_name)
        
        metadata_list.append({
            "topic": topic_name,
            "sentiment": sentiments[i]
        })
        
    logging.info(f"✅ NLP Analysis complete!")
    logging.info(f"   - Discovered {len(unique_topics)} unique topics in the corpus.")
    logging.info(f"   - Sentiment distribution: {sentiments.count('Positive')} Positive, {sentiments.count('Neutral')} Neutral, {sentiments.count('Negative')} Negative.")
    
    return metadata_list