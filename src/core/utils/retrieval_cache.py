import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional


def question_hash(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()


class SparseRetrievalCache:
    """Dictionary-like cache for sparse retrieval results."""

    def __init__(self, cache_data: Dict[str, Any]):
        self.cache_data = cache_data
        self.index_by_qhash = cache_data.get("index_by_qhash", {})
        self.entries = cache_data.get("entries", [])

    @classmethod
    def load(cls, cache_path: str) -> "SparseRetrievalCache":
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def get_by_question(self, question: str) -> Optional[Dict[str, Any]]:
        qh = question_hash(question)
        idx = self.index_by_qhash.get(qh)
        if idx is None:
            return None
        return self.entries[idx]

    @staticmethod
    def save(cache_data: Dict[str, Any], output_path: str):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
