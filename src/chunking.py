from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(texts, chunk_size=300, chunk_overlap=50):
    """
    Splits a list of texts into smaller, overlapping chunks.
    
    :param texts: List of strings (e.g., our cleaned abstracts).
    :param chunk_size: Maximum number of characters per chunk.
    :param chunk_overlap: Number of characters to overlap between chunks 
                          (keeps context from getting cut off).
    :return: List of chunked strings.
    """
    
    # Initialize the recursive splitter
    # It tries to split by paragraphs, then sentences, then words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    
    for text in texts:
        # Skip empty texts (Edge case handling!)
        if not text or not text.strip():
            continue
            
        # Split the text into chunks
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        
    return all_chunks