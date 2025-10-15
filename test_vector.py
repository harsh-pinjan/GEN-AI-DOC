from src.document_loader import load_and_chunk_pdf
from src.vector_store import create_faiss_vector_store, load_faiss_vector_store

# Load chunks from PDF
chunks = load_and_chunk_pdf("data/sample.pdf")

# Create FAISS vector store (now uses HuggingFace embeddings)
vector_store = create_faiss_vector_store(chunks)

# Load it back
vector_store = load_faiss_vector_store()
