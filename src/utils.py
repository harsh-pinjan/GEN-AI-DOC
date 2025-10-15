
import os
import hashlib
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import matplotlib.pyplot as plt
import io

# -------------------------
# PDF utilities
# -------------------------
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for p, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # keep basic page metadata by inserting marker
        text.append(f"\n\n<Page {p+1}>\n{page_text}")
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(text)
    docs = []
    # try to preserve page markers in metadata
    for t in texts:
        # naive page extraction from marker
        page = None
        marker_idx = t.find("<Page")
        if marker_idx != -1:
            # try to parse page number
            import re
            m = re.search(r"<Page\s+(\d+)>", t)
            if m:
                page = int(m.group(1))
        docs.append(Document(page_content=t, metadata={"page": page}))
    return docs

def build_embeddings_and_faiss(docs: List[Document], persist_directory: str) -> FAISS:
    """Build embeddings with HuggingFace and save FAISS locally."""
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, emb)
    os.makedirs(persist_directory, exist_ok=True)
    vector_store.save_local(persist_directory)
    return vector_store

def load_faiss(persist_directory: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # allow dangerous deserialization if index was created locally (explicitly set by you)
    vector_store = FAISS.load_local(persist_directory, emb, allow_dangerous_deserialization=True)
    return vector_store

# -------------------------
# CSV utilities
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def best_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()

def csv_highest(df: pd.DataFrame, col: str) -> Tuple[dict, pd.Series]:
    idx = df[col].idxmax()
    return df.loc[idx].to_dict(), df.loc[idx]

def csv_lowest(df: pd.DataFrame, col: str) -> Tuple[dict, pd.Series]:
    idx = df[col].idxmin()
    return df.loc[idx].to_dict(), df.loc[idx]

def csv_average(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())

# -------------------------
# Chart utilities (matplotlib -> PNG bytes)
# -------------------------
def plot_histogram(df: pd.DataFrame, col: str, title: Optional[str] = None) -> bytes:
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(title or f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def plot_bar_by_group(df: pd.DataFrame, numeric_col: str, group_col: str, title: Optional[str] = None) -> bytes:
    grouped = df.groupby(group_col)[numeric_col].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    grouped.plot.bar(ax=ax)
    ax.set_title(title or f"Mean {numeric_col} by {group_col}")
    ax.set_ylabel(numeric_col)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Helpers
# -------------------------
def file_hash_bytes(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def docs_to_single_hash(docs: List[Document]) -> str:
    joined = "".join([d.page_content for d in docs])
    return hashlib.md5(joined.encode()).hexdigest()

# -------------------------
# Exports
# -------------------------
def export_history_to_csv(history: List[dict], out_path: str):
    rows = []
    for item in history:
        q = item.get("question")
        a = item.get("answer")
        for s in item.get("sources", []):
            rows.append({
                "question": q,
                "answer": a,
                "source_snippet": s.get("snippet"),
                "source_file": s.get("source"),
                "page": s.get("page")
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def export_history_to_txt(history: List[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in history:
            f.write(f"Q: {item.get('question')}\n")
            f.write(f"A: {item.get('answer')}\n")
            f.write("Sources:\n")
            for s in item.get("sources", []):
                f.write(f" - {s.get('source')} (page: {s.get('page')}) snippet: {s.get('snippet')}\n")
            f.write("\n\n")
