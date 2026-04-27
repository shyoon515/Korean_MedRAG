"""
Build dense FAISS index for the main data corpus.

This script mirrors sparse preprocessing:
- corpus source: data/train/corpus/TS_국문*
- chunking: chunk_size=1000, chunk_overlap=100
- encoder: BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_core_components():
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    try:
        from src.core import (
            CorpusLoader,
            DenseRetriever,
            QALoader,
            SparseRetrievalCache,
            question_hash,
            setup_category_loggers,
        )
    except ImportError:
        from core import (
            CorpusLoader,
            DenseRetriever,
            QALoader,
            SparseRetrievalCache,
            question_hash,
            setup_category_loggers,
        )

    return CorpusLoader, DenseRetriever, QALoader, SparseRetrievalCache, question_hash, setup_category_loggers


def collect_queries_by_source(
    qa_loader,
    qa_train_dir: str,
    qa_test_dir: str,
) -> Dict[str, List[Tuple[str, Dict]]]:
    by_source: Dict[str, List[Tuple[str, Dict]]] = {}

    train_root = Path(qa_train_dir)
    test_root = Path(qa_test_dir)

    for split, root in (("train", train_root), ("test", test_root)):
        category_folders = sorted([d for d in root.iterdir() if d.is_dir()])
        for folder in category_folders:
            source_name = folder.name
            qa_list = qa_loader.load_qa_from_folder(folder)
            by_source.setdefault(source_name, [])
            by_source[source_name].extend((split, qa) for qa in qa_list)

    return by_source


def sample_queries(rows: List[Tuple[str, Dict]], queries_per_source: int, seed: int) -> List[Tuple[str, Dict]]:
    if len(rows) <= queries_per_source:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, queries_per_source)


def build_source_entry(split: str, source_name: str, qa: Dict, retrieved: List[Dict]) -> Dict:
    question = qa.get("question", "")
    return {
        "split": split,
        "query_source": source_name,
        "qa_id": qa.get("qa_id"),
        "question": question,
        "num_retrieved": len(retrieved),
        "scores": [item["score"] for item in retrieved],
        "retrieved_items": [
            {
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "score": item["score"],
                "text": item["text"],
            }
            for item in retrieved
        ],
    }


def build_dense_cache(
    corpus_dir: Path,
    qa_train_dir: Path,
    qa_test_dir: Path,
    output_root: Path,
    index_root: Path,
    model_name: str = "BAAI/bge-m3",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    top_k: int = 20,
    max_docs_per_folder: int | None = None,
    max_queries: int | None = None,
    queries_per_source: int = 10,
    sample_seed: int = 42,
    batch_size: int = 64,
) -> Dict[str, Any]:
    (
        CorpusLoader,
        DenseRetriever,
        QALoader,
        SparseRetrievalCache,
        question_hash,
        setup_category_loggers,
    ) = _load_core_components()

    output_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    log_dir = output_root / "logs"
    loggers = setup_category_loggers(str(log_dir), level=logging.INFO, base_name="dense_cache")
    pipeline_logger = loggers["pipeline"]

    pipeline_logger.info("Dense cache build started")
    pipeline_logger.info(
        "Config: model=%s top_k=%s chunk_size=%s chunk_overlap=%s max_docs_per_folder=%s max_queries=%s queries_per_source=%s batch_size=%s corpus_dir=%s output_root=%s index_root=%s",
        model_name,
        top_k,
        chunk_size,
        chunk_overlap,
        max_docs_per_folder,
        max_queries,
        queries_per_source,
        batch_size,
        corpus_dir,
        output_root,
        index_root,
    )

    corpus_loader = CorpusLoader(
        corpus_dir=str(corpus_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs_per_folder=max_docs_per_folder,
        logger=pipeline_logger,
    )
    qa_loader = QALoader(str(qa_train_dir), str(qa_test_dir), logger=pipeline_logger)

    chunks = corpus_loader.load_korean_corpus()

    retriever = DenseRetriever(
        model_name=model_name,
        batch_size=batch_size,
        logger=pipeline_logger,
    )
    retriever.build_index(chunks)

    index_dir = index_root / "data_corpus_index"
    retriever.save_index(str(index_dir))

    by_source = collect_queries_by_source(qa_loader, str(qa_train_dir), str(qa_test_dir))
    source_names = sorted(by_source.keys())
    pipeline_logger.info("Total query sources discovered: %s", len(source_names))

    manifest = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "dataset": "data",
            "index_type": "faiss_dense",
            "encoder": model_name,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "queries_per_source": queries_per_source,
            "sample_seed": sample_seed,
            "schema": "v1-by-source",
        },
        "index": {
            "index_dir": str(index_dir),
            "index_file": "index.faiss",
            "chunks_file": "chunks.jsonl",
            "meta_file": "meta.json",
        },
        "sources": {},
    }

    total_cached_queries = 0
    for source_name in source_names:
        rows = by_source.get(source_name, [])
        sampled_rows = sample_queries(rows, queries_per_source=queries_per_source, seed=sample_seed)
        if max_queries is not None:
            sampled_rows = sampled_rows[:max_queries]

        entries = []
        index_by_qhash = {}

        for split, qa in sampled_rows:
            question = qa.get("question", "")
            retrieved = retriever.search(question, top_k=top_k)
            entry = build_source_entry(split, source_name, qa, retrieved)
            entries.append(entry)
            index_by_qhash[question_hash(question)] = len(entries) - 1

        cache_data = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "retriever": "dense",
                "encoder": model_name,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(chunks),
                "total_queries": len(entries),
                "query_source": source_name,
                "index_dir": str(index_dir),
                "schema": "v1",
            },
            "index_by_qhash": index_by_qhash,
            "entries": entries,
        }

        source_output = output_root / f"{source_name}.json"
        SparseRetrievalCache.save(cache_data, str(source_output))
        total_cached_queries += len(entries)

        manifest["sources"][source_name] = {
            "file": source_output.name,
            "num_cached_queries": len(entries),
            "total_available_queries": len(rows),
        }

        pipeline_logger.info(
            "Source cached(dense): %s cached=%s total_available=%s file=%s",
            source_name,
            len(entries),
            len(rows),
            source_output,
        )

    manifest["meta"]["total_sources"] = len(source_names)
    manifest["meta"]["total_cached_queries"] = total_cached_queries

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    pipeline_logger.info("Dense cache build completed: output=%s index=%s", output_root, index_dir)
    pipeline_logger.info("Manifest saved: %s", manifest_path)
    return manifest


def resolve_output_root(workspace_root: Path, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return workspace_root / "retrieval_cache" / "dense"


def resolve_index_root(workspace_root: Path, index_root: str | None) -> Path:
    if index_root is not None:
        return Path(index_root)
    return workspace_root / "dense_index" / "data"


def find_workspace_root(file_path: Path) -> Path:
    workspace_root = file_path.resolve().parents[1]
    if not (workspace_root / "data").exists():
        raise RuntimeError(f"Workspace data directory not found under {workspace_root}")
    return workspace_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dense retrieval cache by query source")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--index-root", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="BAAI/bge-m3")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs-per-folder", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--queries-per-source", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    workspace_root = find_workspace_root(Path(__file__))
    corpus_dir = workspace_root / "data" / "train" / "corpus"
    qa_train_dir = workspace_root / "data" / "train" / "qa"
    qa_test_dir = workspace_root / "data" / "test" / "qa"
    output_root = resolve_output_root(workspace_root, args.output_root)
    index_root = resolve_index_root(workspace_root, args.index_root)

    build_dense_cache(
        corpus_dir=corpus_dir,
        qa_train_dir=qa_train_dir,
        qa_test_dir=qa_test_dir,
        output_root=output_root,
        index_root=index_root,
        model_name=args.encoder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_docs_per_folder=args.max_docs_per_folder,
        max_queries=args.max_queries,
        queries_per_source=args.queries_per_source,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
    )
