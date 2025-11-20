import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

CSV_PATH = "./data/ipl_with_summary.csv"

# Load data
df = pd.read_csv(CSV_PATH).dropna(subset=["Player"]).reset_index(drop=True)

def row_to_text(row):
    lines = []
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            continue
        lines.append(f"{col}: {val}")
    return "\n".join(lines)

# Build docs
docs = []
for _, row in df.iterrows():
    full = row_to_text(row)
    docs.append(Document(page_content=full, metadata={"player": row["Player"]}))

print("Documents:", len(docs))

# Split docs
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))

# Embeddings
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Save FAISS
if not os.path.exists("faiss_index"):
    vectorstore = FAISS.from_documents(chunks, emb)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved!")
else:
    print("FAISS index already exists.")
