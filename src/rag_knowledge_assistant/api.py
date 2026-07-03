"""FastAPI interface: a streaming answer endpoint + a minimal chat UI with citations.

Endpoints:
  GET /            -> chat UI (single self-contained page)
  GET /health      -> health check
  GET /ask?q=...   -> full answer as JSON (convenient + testable)
  GET /stream?q=.. -> Server-Sent Events: answer word-by-word, then a citations event

The extractive answer is computed then streamed word-by-word to exercise the SSE
plumbing and incremental UI rendering; a ClaudeGenerator would stream tokens natively.

Run with:  uv run uvicorn rag_knowledge_assistant.api:app --reload
"""

import json
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from rag_knowledge_assistant.generation import Answer, generate_answer
from rag_knowledge_assistant.retrieval import Retriever, load_retriever

app = FastAPI(title="RAG Knowledge Assistant")


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return load_retriever()


def _answer(query: str) -> Answer:
    return generate_answer(get_retriever(), query)


def _payload(answer: Answer) -> dict:
    return {
        "query": answer.query,
        "text": answer.text,
        "refused": answer.refused,
        "citations": [c._asdict() for c in answer.citations],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str) -> dict:
    return _payload(_answer(q))


@app.get("/stream")
def stream(q: str) -> StreamingResponse:
    answer = _answer(q)

    def events():
        for word in answer.text.split():
            yield f"data: {word}\n\n"
        citations = [c._asdict() for c in answer.citations]
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG Knowledge Assistant</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.3rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .6rem; font-size: 1rem; }
  button { padding: .6rem 1rem; font-size: 1rem; cursor: pointer; }
  #answer { margin: 1.2rem 0; line-height: 1.6; min-height: 2rem; white-space: pre-wrap; }
  #answer.refused { color: #b45309; font-style: italic; }
  .cite { font-size: .8rem; color: #2563eb; vertical-align: super; }
  #sources { list-style: none; padding: 0; border-top: 1px solid #8883; margin-top: 1rem; }
  #sources li { padding: .35rem 0; font-size: .9rem; }
  #sources a { color: #2563eb; text-decoration: none; }
  .hint { color: #8888; font-size: .85rem; }
</style>
</head>
<body>
  <h1>📚 RAG Knowledge Assistant</h1>
  <p class="hint">Ask about ML/AI topics (overfitting, transformers, cosine similarity,
    ...). Answers cite their sources.</p>
  <form id="f">
    <input id="q" placeholder="What is overfitting?" autocomplete="off" autofocus/>
    <button type="submit">Ask</button>
  </form>
  <div id="answer"></div>
  <ul id="sources"></ul>
<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const answer = document.getElementById('answer');
const sources = document.getElementById('sources');
let es;

f.addEventListener('submit', (e) => {
  e.preventDefault();
  if (!q.value.trim()) return;
  if (es) es.close();
  answer.textContent = '';
  answer.className = '';
  sources.innerHTML = '';
  es = new EventSource('/stream?q=' + encodeURIComponent(q.value));
  es.onmessage = (ev) => { answer.textContent += ev.data + ' '; };
  es.addEventListener('citations', (ev) => {
    const cites = JSON.parse(ev.data);
    if (cites.length === 0) { answer.className = 'refused'; }
    for (const c of cites) {
      const li = document.createElement('li');
      li.innerHTML = '<span class="cite">[' + c.marker + ']</span> ' +
        '<a href="' + c.source_url + '" target="_blank" rel="noopener">' + c.title + '</a>' +
        ' <span class="hint">(score ' + c.score.toFixed(3) + ')</span>';
      sources.appendChild(li);
    }
  });
  es.addEventListener('done', () => es.close());
  es.onerror = () => es.close();
});
</script>
</body>
</html>
"""
