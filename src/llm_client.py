import os
import time
import logging
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Configure logging for clean console output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Initialize the Groq client using the API key from the .env file
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Define the strict system prompt to prevent hallucinations
SYSTEM_PROMPT = """You are an expert AI research assistant. 
Your task is to answer the user's question using ONLY the provided context.
Rules:
1. If the answer is clearly in the context, provide a concise and accurate response.
2. If the answer is NOT in the context, or if the question is completely unrelated to the context (out-of-domain), you MUST reply exactly with: "I cannot answer this based on the provided documents."
3. Do not use outside knowledge. Do not make things up."""

def generate_answer(context: str, query: str) -> dict:
    """
    Sends the context and query to the LLM and returns the generated answer.
    Includes robust error handling (timeouts, rate limits) and latency logging.
    """
    # Inject the context into the user prompt
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    start_time = time.time()
    
    try:
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-safeguard-20b", # Fast, free, and highly reliable model
            temperature=0.1, # Low temperature ensures factual, deterministic answers
            max_tokens=512,
            timeout=30 # 30-second timeout to prevent the script from hanging
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        answer = chat_completion.choices[0].message.content
        
        logging.info(f"✅ LLM Generation complete! Latency: {latency_ms:.2f} ms")
        
        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "error": None
        }

    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        error_msg = f"API Error: {str(e)}"
        logging.error(f"❌ {error_msg} (Latency: {latency_ms:.2f} ms)")
        
        return {
            "answer": None,
            "latency_ms": latency_ms,
            "error": error_msg
        }