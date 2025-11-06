import os
import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/text-embedding-004"  # Gemini embedding model
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    chunks = []
    for d in docs:
        if getattr(d, "text", None):
            chunks.extend(splitter.split_text(d.text))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(model=EMBED_MODEL, content=text)
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            embeddings.append([])
    return embeddings
