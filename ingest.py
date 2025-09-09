import os, glob, uuid
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY missing in .env"

DB_PATH = "chroma_db"
COLLECTION_NAME = "support_kb"

# ---------- simple text chunking ----------
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def simple_chunk(text: str, max_chars=800, overlap=100) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
                tail = buf[-overlap:] if len(buf) > overlap else buf
                buf = (tail + "\n\n" + p).strip()
            else:
                chunks.append(p[:max_chars])
                buf = p[max_chars-overlap:max_chars]
    if buf:
        chunks.append(buf)
    return chunks

def load_docs(folder="knowledge"):
    files = glob.glob(os.path.join(folder, "**", "*.md"), recursive=True) \
          + glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    docs: List[Chunk] = []
    for path in files:
        text = read_text(path)
        title = os.path.basename(path)
        for i, chunk_text in enumerate(simple_chunk(text)):
            cid = f"{title}-{i}-{uuid.uuid4().hex[:8]}"
            docs.append(Chunk(id=cid, text=chunk_text, source=path, title=title))
    return docs

def main():
    print("Loading docs...")
    docs = load_docs()
    if not docs:
        print("No docs found in 'knowledge/'")
        return

    print(f"Loaded {len(docs)} chunks")

    print("Connecting to Chroma (vector DB)...")
    client = chromadb.PersistentClient(path=DB_PATH)

    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    # create or open collection
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # wipe previous data safely
    try:
        existing = col.get()
        if existing and existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    print("Adding chunks...")
    col.add(
        ids=[c.id for c in docs],
        documents=[c.text for c in docs],
        metadatas=[{"source": c.source, "title": c.title} for c in docs],
    )
    print("Done. ✅")

if __name__ == "__main__":
    main()
