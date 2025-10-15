from src.qa_chain import build_qa_chain

qa_chain = build_qa_chain()

query = "What is Artificial Intelligence?"
response = qa_chain.invoke({"query": query})

print("Q:", query)
print("A:", response["result"])
print("\n--- Sources ---")
for doc in response["source_documents"]:
    print(doc.page_content[:200])  # print snippet of source text
