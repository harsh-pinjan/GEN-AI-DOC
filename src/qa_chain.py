from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from src.vector_store import load_faiss_vector_store

def build_qa_chain():
    # Load FAISS
    vector_store = load_faiss_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Load local HuggingFace model
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain
