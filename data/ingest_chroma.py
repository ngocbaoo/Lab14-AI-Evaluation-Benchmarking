import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv


def load_chunks(chunks_path: Path) -> Tuple[List[str], List[str], List[Dict]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
            documents.append(obj["text"])
            metadatas.append(obj.get("metadata") or {})
    return ids, documents, metadatas


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest `data/chunks.jsonl` into a local persistent Chroma DB.")
    parser.add_argument("--db-path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "lab14"))
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection before ingest")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    args = parser.parse_args()

    load_dotenv()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"Missing chunks file: {chunks_path} (run data/chunk_doc.py first)")

    # Lazy imports so this script can be present even before deps are installed.
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=args.db_path)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=args.embedding_model)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=args.collection, embedding_function=embedding_fn)
    ids, documents, metadatas = load_chunks(chunks_path)

    total = len(ids)
    if total == 0:
        raise SystemExit("No chunks found to ingest.")

    for start in range(0, total, args.batch_size):
        end = min(total, start + args.batch_size)
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"✅ Ingested {total} chunks into Chroma.")
    print(f"db_path={args.db_path}")
    print(f"collection={args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

