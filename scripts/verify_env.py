# parallax-rag-knowledge-extractor/
# ├── data/
# │   ├── raw/             <-- Put your 5,000+ downloaded documents here
# │   └── processed/       <-- Your final clean file will go here
# ├── src/
# │   └── preprocessing.py <-- Your text cleaning code goes here
# ├── tests/
# │   └── test_preprocessing.py <-- Your unit tests go here
# ├── scripts/
# │   └── verify_env.py    <-- Your environment check script goes here
# ├── .gitignore           <-- MUST ignore /data/ and .env files
# ├── requirements.txt     <-- Update this with all needed libraries
# ── README.md            <-- Keep updating this!

import sys

def check_cuda():
    """Check if GPU/CUDA is available."""
    print("Checking GPU/CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA is not available. The project will run on CPU.")
    except ImportError:
        print("❌ PyTorch is not installed. (Required for sentence-transformers)")

def check_libraries():
    """Verify all required libraries are installed and importable."""
    print("\nChecking required libraries...")
    
    libraries = [
        ("pandas", "pandas"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("spacy", "spacy")
    ]
    
    all_success = True
    for module_name, pip_name in libraries:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully.")
        except ImportError:
            print(f"❌ Failed to import {module_name}. Run: pip install {pip_name}")
            all_success = False
            
    # Bonus: Check if the required spaCy model is downloaded
    print("\nChecking spaCy language model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        all_success = False

    return all_success

if __name__ == "__main__":
    print("="*40)
    print("  Environment Verification Script")
    print("="*40)
    
    check_cuda()
    success = check_libraries()
    
    print("\n" + "="*40)
    if success:
        print(" SUCCESS: Environment is fully configured!")
    else:
        print("⚠️ ISSUES FOUND: Please fix the missing dependencies above.")
    print("="*40)