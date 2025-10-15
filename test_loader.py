from src.document_loader import load_and_chunk_pdf

chunks = load_and_chunk_pdf("data/sample.pdf")
print(f"Total chunks: {len(chunks)}")
print(chunks[0].page_content[:500])  # print first 500 chars of first chunk
