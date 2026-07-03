"""Load the document corpus (Wikipedia plain-text extracts) with attribution."""

import json
from pathlib import Path
from typing import NamedTuple

CORPUS_DIR = Path("data/corpus")
SOURCES_PATH = Path("data/corpus_sources.json")


class Document(NamedTuple):
    doc_id: str
    title: str
    source_url: str
    text: str


def load_corpus(corpus_dir: Path = CORPUS_DIR, sources_path: Path = SOURCES_PATH) -> list[Document]:
    """Read every ``.txt`` in the corpus dir, enriched with manifest metadata."""
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        doc_id = path.stem
        meta = sources.get(doc_id, {})
        docs.append(
            Document(
                doc_id=doc_id,
                title=meta.get("title", doc_id),
                source_url=meta.get("url", ""),
                text=path.read_text().strip(),
            )
        )
    return docs
