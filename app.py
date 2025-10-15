# app.py
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.schema import Document
import pandas as pd
import re
from src.utils import (
    extract_text_from_pdf, chunk_text, build_embeddings_and_faiss, load_faiss,
    load_csv, best_numeric_columns, csv_highest, csv_lowest, csv_average,
    plot_histogram, plot_bar_by_group, file_hash_bytes, docs_to_single_hash,
    export_history_to_csv, export_history_to_txt
)

# ------------- Page config -------------
st.set_page_config(page_title="PRO: PDF+CSV Q&A", layout="wide")
st.title(" PRO PDF + CSV Q&A — Advanced")

# ------------- Session init -------------
if "vectorstore_map" not in st.session_state:
    st.session_state.vectorstore_map = {}  # persist_dir -> FAISS
if "csv_map" not in st.session_state:
    st.session_state.csv_map = {}         # filename -> DataFrame
if "history" not in st.session_state:
    st.session_state.history = []         # list of {question, answer, sources}
if "llm_name" not in st.session_state:
    st.session_state.llm_name = "google/flan-t5-small"
if "llm_obj" not in st.session_state:
    st.session_state.llm_obj = None

# ------------- Sidebar: settings & model -------------
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose local HF model", ["google/flan-t5-small", "t5-base"], key="model_choice_1")
    # Lazy load model on demand
    if model_choice != st.session_state.llm_name or st.session_state.llm_obj is None:
        st.info("Loading/Refreshing local model (may take time on first load)...")
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        st.session_state.llm_obj = HuggingFacePipeline(pipeline=pipe)
        st.session_state.llm_name = model_choice
    st.markdown("---")
    st.write("Export / Downloads")
    os.makedirs("outputs", exist_ok=True)
    if st.button("Export history → CSV", key="export_csv_btn"):
        export_history_to_csv(st.session_state.history, "outputs/qa_history.csv")
        st.success("Saved outputs/qa_history.csv")
    if st.button("Export history → TXT", key="export_txt_btn"):
        export_history_to_txt(st.session_state.history, "outputs/qa_history.txt")
        st.success("Saved outputs/qa_history.txt")

# ------------- Main: Tabs -------------
tab1, tab2, tab3 = st.tabs([" Upload Files", " Ask & Browse", " CSV Analytics"])

# ---------- TAB 1: Upload Files ----------
with tab1:
    st.subheader("Upload PDFs and CSVs (multi-select supported)")
    uploaded = st.file_uploader("Drag & drop files", type=["pdf", "csv"], accept_multiple_files=True, key="uploader_advanced")
    if uploaded:
        for f in uploaded:
            b = f.read()
            h = file_hash_bytes(b)
            # save raw file to data/
            os.makedirs("data", exist_ok=True)
            path = os.path.join("data", f.name)
            with open(path, "wb") as fh:
                fh.write(b)

            if f.name.lower().endswith(".pdf"):
                st.info(f"Processing PDF: {f.name}")
                text = extract_text_from_pdf(path)
                docs = chunk_text(text)
                persist_dir = f"faiss_index_{h}"
                # only build if not already in map
                if persist_dir not in st.session_state.vectorstore_map:
                    vs = build_embeddings_and_faiss(docs, persist_dir)
                    st.session_state.vectorstore_map[persist_dir] = {"store": vs, "docs_hash": docs_to_single_hash(docs), "source": f.name}
                    st.success(f"PDF indexed: {f.name}")
                else:
                    st.info(f"Already indexed: {f.name}")

            elif f.name.lower().endswith(".csv"):
                st.info(f"Loading CSV: {f.name}")
                df = load_csv(path)
                st.session_state.csv_map[f.name] = df
                st.success(f"CSV loaded: {f.name}")

    st.markdown("---")
    st.write("Indexed sources:")
    for k, v in st.session_state.vectorstore_map.items():
        st.write(f"- {v['source']} (index: {k})")
    for name in st.session_state.csv_map.keys():
        st.write(f"- CSV: {name}")

# ---------- TAB 2: Ask & Browse ----------
with tab2:
    st.subheader("Ask questions (will query CSVs with structured logic and PDFs with semantic retrieval)")
    user_q = st.text_input("Type your question here", key="ask_input_1")

    if st.button("Ask", key="ask_button_1") and user_q:
        answer_parts = []
        sources_out = []

        # 1) CSV structured handling
        csv_keywords = ["highest", "lowest", "max", "min", "sum", "average", "mean", "count", "total"]
        csv_trigger = any(k in user_q.lower() for k in csv_keywords) and len(st.session_state.csv_map) > 0
        if csv_trigger:
            for name, df in st.session_state.csv_map.items():
                try:
                    num_cols = best_numeric_columns(df)
                    if "highest" in user_q.lower() or "max" in user_q.lower():
                        for col in num_cols:
                            row_dict, row_series = csv_highest(df, col)
                            # try to use a human name column if present
                            display = row_dict.get("Name") or row_dict.get("name") or row_dict.get(next(iter(df.columns), ""), "")
                            answer_parts.append(f"CSV `{name}` — Highest {col}: {display} ({row_dict.get(col)})")
                            sources_out.append({"snippet": str(row_series.to_dict()), "source": name, "page": None})
                    elif "lowest" in user_q.lower() or "min" in user_q.lower():
                        for col in num_cols:
                            row_dict, row_series = csv_lowest(df, col)
                            display = row_dict.get("Name") or row_dict.get("name") or ""
                            answer_parts.append(f"CSV `{name}` — Lowest {col}: {display} ({row_dict.get(col)})")
                            sources_out.append({"snippet": str(row_series.to_dict()), "source": name, "page": None})
                    elif "average" in user_q.lower() or "mean" in user_q.lower():
                        for col in num_cols:
                            avg = csv_average(df, col)
                            answer_parts.append(f"CSV `{name}` — Average {col}: {avg:.2f}")
                            sources_out.append({"snippet": f"avg {col}: {avg:.2f}", "source": name, "page": None})
                except Exception as e:
                    answer_parts.append(f"Error analyzing CSV `{name}`: {e}")

        # 2) PDF semantic retrieval (query all indexed FAISS stores)
        # Only run PDF query if not purely CSV-trigger (so we avoid redundant pdf answers on numeric CSV queries)
        if not (csv_trigger and len(st.session_state.csv_map) > 0):
            # aggregate answers from all indexes
            for persist_dir, meta in st.session_state.vectorstore_map.items():
                vs = meta["store"]
                retriever = vs.as_retriever(search_kwargs={"k": 3})
                qa = RetrievalQA.from_chain_type(llm=st.session_state.llm_obj, retriever=retriever, chain_type="stuff", return_source_documents=True)
                try:
                    res = qa({"query": user_q})
                    txt = res.get("result", "")
                    if txt:
                        answer_parts.append(f"PDF `{meta['source']}` — {txt}")
                    for d in res.get("source_documents", []):
                        sources_out.append({
                            "snippet": d.page_content[:200],
                            "source": meta["source"],
                            "page": d.metadata.get("page")
                        })
                except Exception as e:
                    answer_parts.append(f"PDF `{meta['source']}` error: {e}")

        final_answer = "\n\n".join(answer_parts) if answer_parts else "No answer found."

        # save history structured
        st.session_state.history.append({"question": user_q, "answer": final_answer, "sources": sources_out})
        st.success("Answer ready — see below")

    # show last answer
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("Latest Answer")
        st.markdown(latest["answer"])
        st.markdown("**Sources:**")
        for s in latest["sources"]:
            st.markdown(f"- {s.get('source')} (page: {s.get('page')}) — {s.get('snippet')[:200]}")

# ---------- TAB 3: CSV Analytics ----------
with tab3:
    st.subheader("CSV Analytics & Charts")
    if not st.session_state.csv_map:
        st.info("Upload CSV files in the Upload Files tab to enable analytics.")
    else:
        sel_name = st.selectbox("Select CSV to analyze", list(st.session_state.csv_map.keys()), key="csv_select_1")
        df = st.session_state.csv_map.get(sel_name)
        st.write("Preview:")
        st.dataframe(df.head(10))

        numeric_cols = best_numeric_columns(df)
        st.markdown("**Numeric columns detected:** " + ", ".join(numeric_cols) if numeric_cols else "No numeric columns detected")

        if numeric_cols:
            col = st.selectbox("Choose numeric column", numeric_cols, key="col_select_1")
            st.markdown("Distribution:")
            img_bytes = plot_histogram(df, col, title=f"Distribution of {col} in {sel_name}")
            st.image(img_bytes)

            # group by selection
            group_cols = [c for c in df.columns if df[c].dtype == object]
            if group_cols:
                grp = st.selectbox("Group by (optional):", ["None"] + group_cols, key="group_select_1")
                if grp != "None":
                    img2 = plot_bar_by_group(df, col, grp, title=f"Mean {col} by {grp}")
                    st.image(img2)

# ------------- Bottom: quick history viewer -------------
st.markdown("---")
st.subheader("Session Q&A History")
for i, item in enumerate(reversed(st.session_state.history[-20:])):
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")
    st.markdown("---")
