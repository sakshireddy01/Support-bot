import os, json
from typing import List, Tuple
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ----- config -----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY missing in .env"

DB_PATH = "chroma_db"
COLLECTION_NAME = "support_kb"
MODEL = "gpt-4o-mini"   # you can switch to gpt-4o if you want

# ----- init clients -----
client = OpenAI()  # reads key from env
chroma = chromadb.PersistentClient(path=DB_PATH)
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
col = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn
)

# ----- FastAPI app -----
app = FastAPI(title="Support Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")

class AskBody(BaseModel):
    question: str

PROMPT = """You are a careful customer support assistant.
Answer ONLY using the Context. If the answer is missing or uncertain, say:
"I'm not fully sure based on our docs. I've flagged this for a human specialist."

Always include bracketed citations like [1], [2] that refer to the "Context Sources".
Return a JSON object: {{"answer": "...", "citations": [1,2], "confidence": 0.0-1.0}}

Question: {question}

Context:
{context}

Context Sources:
{sources}
"""

def build_context(results) -> Tuple[str, List[dict]]:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    sources = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        title = m.get("title", "doc")
        src = m.get("source", "")
        blocks.append(f"[{i}] {d}")
        sources.append({"n": i, "title": title, "url": src})
    return "\n\n---\n\n".join(blocks), sources

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
def ask(body: AskBody):
    q = body.question.strip()
    if not q:
        return JSONResponse({"answer": "Please type a question.", "citations": [], "confidence": 0.0})

    # 1) retrieve top chunks
    hits = col.query(query_texts=[q], n_results=4)

    # 2) build prompt
    context, sources = build_context(hits)
    sources_str = "\n".join([f"[{s['n']}] {s['title']} - {s['url']}" for s in sources])
    prompt = PROMPT.format(question=q, context=context, sources=sources_str)

    # 3) call LLM
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )
    data = json.loads(resp.choices[0].message.content)

    # 4) pretty citations for UI
    used = set(data.get("citations", []))
    pretty = [s for s in sources if s["n"] in used]

    return JSONResponse({
        "answer": data.get("answer", "(no answer)"),
        "citations": pretty,
        "confidence": data.get("confidence", 0.0)
    })
