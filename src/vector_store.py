from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_faiss_vector_store(chunks, persist_directory="faiss_index"):
    """
    Takes document chunks, creates embeddings, and stores in FAISS
    """

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    
    vector_store = FAISS.from_documents(chunks, embeddings)

   
    vector_store.save_local(persist_directory)
    print(f"FAISS vector store created and saved at {persist_directory}")
    
    return vector_store

def load_faiss_vector_store(persist_directory="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS vector store loaded from {persist_directory}")
    return vector_store
