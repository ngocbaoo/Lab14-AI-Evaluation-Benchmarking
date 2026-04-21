import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks(chunks_jsonl: Path) -> List[Dict]:
    chunks: List[Dict] = []
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                {
                    "id": obj["id"],
                    "text": obj["text"],
                    "norm": _normalize(obj["text"]),
                }
            )
    return chunks


_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _jaccard(a: List[str], b: List[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)

def _overlap_ratio(context_tokens: List[str], chunk_tokens: List[str]) -> float:
    ctx_set = set(context_tokens)
    chk_set = set(chunk_tokens)
    if not ctx_set or not chk_set:
        return 0.0
    return len(ctx_set & chk_set) / len(ctx_set)


def find_expected_ids(context: str, chunks: List[Dict]) -> Tuple[List[str], Optional[str]]:
    if not context or not isinstance(context, str):
        return [], "missing/invalid `context`"

    needle = _normalize(context)
    if not needle:
        return [], "empty `context`"

    hits = [c["id"] for c in chunks if needle in c["norm"]]
    if hits:
        return hits[:3], None

    # Fallback: approximate match by token overlap.
    ctx_tokens = _tokenize(needle)
    best_id = None
    best_score = 0.0
    for c in chunks:
        score = _overlap_ratio(ctx_tokens, _tokenize(c["norm"]))
        if score > best_score:
            best_score = score
            best_id = c["id"]

    if best_id is not None and best_score >= 0.60:
        return [best_id], f"approx_match(overlap={best_score:.3f})"

    return [], "context not found in any chunk (check chunking strategy / source doc mismatch)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach expected_retrieval_ids to golden_set.jsonl from context->chunk match.")
    parser.add_argument("--golden", default="data/golden_set.jsonl", help="Input golden set JSONL")
    parser.add_argument("--chunks", default="data/chunks.jsonl", help="Chunks JSONL (from chunk_doc.py)")
    parser.add_argument("--output", default="data/golden_set.with_expected_ids.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    chunks_path = Path(args.chunks)
    out_path = Path(args.output)
    if not golden_path.exists():
        raise SystemExit(f"Missing: {golden_path}")
    if not chunks_path.exists():
        raise SystemExit(f"Missing: {chunks_path} (run chunk_doc.py first)")

    chunks = load_chunks(chunks_path)
    updated = 0
    errors = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            expected_ids, err = find_expected_ids(obj.get("context", ""), chunks)
            if expected_ids or err in {"missing/invalid `context`", "empty `context`"}:
                # For "out-of-context" cases the dataset may intentionally omit context.
                # Represent this as no relevant docs.
                obj["expected_retrieval_ids"] = expected_ids
                if err in {"missing/invalid `context`", "empty `context`"}:
                    obj.setdefault("metadata", {})
                    if isinstance(obj["metadata"], dict):
                        obj["metadata"]["expected_retrieval_ids_note"] = "no_context"
                updated += 1
            else:
                errors += 1
                obj.setdefault("metadata", {})
                if isinstance(obj["metadata"], dict):
                    obj["metadata"]["expected_retrieval_ids_error"] = err
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote -> {out_path}")
    print(f"attached_expected_ids: {updated}")
    print(f"missing_expected_ids: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
