# Text cleaning and chunking

import re

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Split text into chunks 
def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks