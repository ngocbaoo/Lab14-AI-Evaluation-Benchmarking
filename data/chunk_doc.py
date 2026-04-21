import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]

def chunk_paragraphs(paragraphs: Iterable[str], *, max_chars: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if not buf:
            buf.append(paragraph)
            buf_len = paragraph_len
            continue

        if buf_len + 2 + paragraph_len <= max_chars:
            buf.append(paragraph)
            buf_len += 2 + paragraph_len
            continue

        chunks.append("\n\n".join(buf).strip())
        buf = [paragraph]
        buf_len = paragraph_len

    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


def build_chunks(doc_path: Path, *, max_chars: int) -> List[Dict]:
    raw = doc_path.read_text(encoding="utf-8")
    normalized = _normalize_text(raw)
    paragraphs = _split_paragraphs(normalized)
    chunks_text = chunk_paragraphs(paragraphs, max_chars=max_chars)

    chunks: List[Dict] = []
    for i, text in enumerate(chunks_text, 1):
        chunk_id = f"{doc_path.name}::chunk_{i:05d}"
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": {"source": str(doc_path), "chunk_index": i},
            }
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk a document into deterministic chunk IDs (for Chroma ingestion).")
    parser.add_argument("--input", default="data.md", help="Input document path (default: data.md)")
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output chunks JSONL (default: data/chunks.jsonl)")
    parser.add_argument("--max-chars", type=int, default=1200, help="Max characters per chunk (default: 1200)")
    args = parser.parse_args()

    doc_path = Path(args.input)
    out_path = Path(args.output)
    if not doc_path.exists():
        raise SystemExit(f"Missing input file: {doc_path}")

    chunks = build_chunks(doc_path, max_chars=args.max_chars)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

