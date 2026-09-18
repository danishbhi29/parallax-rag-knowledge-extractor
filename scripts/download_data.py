import os
import json
from datasets import load_dataset

def download_and_save_data():
    # 1. Ensure the raw data directory exists
    os.makedirs("data/raw", exist_ok=True)
    print("Downloading 5,000 ArXiv papers... This may take a minute.")
    
    # 2. Load 5,000 documents from the ArXiv dataset on Hugging Face
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:5000]")
    
    # 3. Save the data as a JSONL file (standard format for AI pipelines)
    output_path = "data/raw/raw_documents.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(dataset):
            # We use the index as 'id', and safely get 'title' and 'abstract'
            clean_item = {
                "id": str(idx),
                "title": item.get("title", "No Title"),
                "abstract": item.get("abstract", "No Abstract")
            }
            f.write(json.dumps(clean_item) + "\n")
            
    print(f"✅ Success! Saved 5,000 documents to {output_path}")

if __name__ == "__main__":
    download_and_save_data()