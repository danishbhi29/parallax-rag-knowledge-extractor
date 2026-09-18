import os
import json
import sys

# This ensures Python can find your 'src' folder to import the cleaning functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the functions we built and tested
from preprocessing import clean_text, is_english

def process_dataset():
    # Define paths for raw input and processed output
    raw_path = "data/raw/raw_documents.jsonl"
    processed_path = "data/processed/clean_corpus.jsonl"
    
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starting data cleaning pipeline on 5,000 documents...")
    cleaned_count = 0
    skipped_count = 0
    
    # Open the raw file to read, and the processed file to write
    with open(raw_path, "r", encoding="utf-8") as infile, \
         open(processed_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            # Load the JSON document
            doc = json.loads(line)
            
            # Clean the title and abstract using our pipeline
            clean_title = clean_text(doc.get("title", ""))
            clean_abstract = clean_text(doc.get("abstract", ""))
            
            # Language filter: only keep the document if the abstract is English
            if not is_english(clean_abstract):
                skipped_count += 1
                continue
            
            # Create the final cleaned document
            cleaned_doc = {
                "id": doc["id"],
                "title": clean_title,
                "abstract": clean_abstract
            }
            
            # Write the cleaned document to the new file
            outfile.write(json.dumps(cleaned_doc) + "\n")
            cleaned_count += 1
            
    print(f"\n✅ Pipeline Complete!")
    print(f"   - Successfully processed & saved: {cleaned_count} documents")
    print(f"   - Skipped (Non-English or empty): {skipped_count} documents")
    print(f"   - Output saved to: {processed_path}")

if __name__ == "__main__":
    process_dataset()