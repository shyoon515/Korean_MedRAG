"""Dense embedding retriever backed by FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers not installed")
    SentenceTransformer = None

from .base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense embedding retriever (FAISS + SentenceTransformer)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        logger=None,
    ):
        """Initialize retriever.

        Args:
            model_name: Hugging Face sentence-transformers model id.
            batch_size: Embedding batch size used during indexing/search.
            logger: Optional logger instance.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger = logger
        self.model = None
        self.chunks = None
        self.index = None
        self.embedding_dim = None
        self._init_model()

    def _init_model(self):
        """Load sentence-transformer model."""
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers not installed")

        if self.logger:
            self.logger.info("Loading dense model: %s", self.model_name)
        else:
            print(f"Loading model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        if self.logger:
            self.logger.info(
                "Dense model loaded on device=%s embedding_dim=%s",
                self.model.device,
                self.embedding_dim,
            )
        else:
            print(f"Model loaded on device: {self.model.device}")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunk list."""
        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]
        if self.logger:
            self.logger.info("Dense embedding started: total_chunks=%s", len(texts))
        else:
            print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        if embeddings.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

        self.embedding_dim = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        if self.logger:
            self.logger.info(
                "Dense index built: chunks=%s dim=%s index_size=%s",
                len(chunks),
                self.embedding_dim,
                self.index.ntotal,
            )
        else:
            print(f"Dense index built with {len(chunks)} chunks (dim={self.embedding_dim})")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run dense retrieval with cosine-like similarity (inner product)."""
        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        if self.index.ntotal == 0:
            return []

        k = min(max(top_k, 1), self.index.ntotal)
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(np.ascontiguousarray(query_embedding), k)

        scores = scores[0]

        results = []
        for rank, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            similarity = scores[rank]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(similarity),
                "rank": rank + 1,
                "doc_id": chunk.get("doc_id"),
                "source": chunk.get("source"),
                "retriever": "Dense"
            })

        return results

    def save_index(self, index_dir: str):
        """Persist FAISS index and chunk metadata to disk."""
        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = out_dir / "index.faiss"
        chunks_path = out_dir / "chunks.jsonl"
        meta_path = out_dir / "meta.json"

        faiss.write_index(self.index, str(faiss_path))

        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        meta = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "total_chunks": len(self.chunks),
            "faiss_metric": "inner_product",
            "normalized_embeddings": True,
            "index_file": faiss_path.name,
            "chunks_file": chunks_path.name,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.logger:
            self.logger.info("Dense index saved: %s", out_dir)
        else:
            print(f"Dense index saved: {out_dir}")

    def load_index(self, index_dir: str):
        """Load FAISS index and chunk metadata from disk."""
        base = Path(index_dir)
        faiss_path = base / "index.faiss"
        chunks_path = base / "chunks.jsonl"
        meta_path = base / "meta.json"

        if not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Dense index files not found under {base}")

        self.index = faiss.read_index(str(faiss_path))
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.embedding_dim = int(meta.get("embedding_dim", self.index.d))
        else:
            self.embedding_dim = int(self.index.d)

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/chunk size mismatch: index={self.index.ntotal} chunks={len(self.chunks)}"
            )

        if self.logger:
            self.logger.info("Dense index loaded: %s (size=%s)", base, self.index.ntotal)
        else:
            print(f"Dense index loaded: {base} (size={self.index.ntotal})")
