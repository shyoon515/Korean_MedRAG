"""
Build sparse BM25 retrieval caches for KorMedMCQA splits.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

try:
    from src.core import (
        CorpusLoader,
        BM25Retriever,
        KorMedMCQALoader,
        SparseRetrievalCache,
        build_retrieval_query,
        question_hash,
        setup_category_loggers,
    )
except ImportError:
    from core import (
        CorpusLoader,
        BM25Retriever,
        KorMedMCQALoader,
        SparseRetrievalCache,
        build_retrieval_query,
        question_hash,
        setup_category_loggers,
    )


KORMEDMCQA_SPLITS = [
    "dentist_train",
    "dentist_dev",
    "dentist_test",
    "doctor_train",
    "doctor_dev",
    "doctor_test",
]


def build_cache_entry(
    split_name: str,
    qa: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    retrieval_query: str,
) -> Dict[str, Any]:
    return {
        "dataset": "KorMedMCQA",
        "split": split_name,
        "source": split_name,
        "question_id": qa.get("question_id"),
        "question": qa.get("question", ""),
        "retrieval_query": retrieval_query,
        "subject": qa.get("subject"),
        "year": qa.get("year"),
        "period": qa.get("period"),
        "q_number": qa.get("q_number"),
        "options": qa.get("options", {}),
        "answer": qa.get("answer"),
        "cot": qa.get("cot", ""),
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


def build_kormedmcqa_cache(
    workspace_root: Path,
    dataset_root: Path,
    corpus_dir: Path,
    output_root: Path,
    top_k: int = 20,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    max_docs_per_folder: int | None = None,
    split_specs: List[str] | None = None,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    log_dir = output_root / "logs"
    loggers = setup_category_loggers(str(log_dir), level=logging.INFO, base_name="kormedmcqa_sparse_cache")
    pipeline_logger = loggers["pipeline"]

    pipeline_logger.info("KorMedMCQA sparse cache build started")
    pipeline_logger.info(
        "Config: top_k=%s chunk_size=%s chunk_overlap=%s max_docs_per_folder=%s dataset_root=%s output_root=%s",
        top_k,
        chunk_size,
        chunk_overlap,
        max_docs_per_folder,
        dataset_root,
        output_root,
    )

    corpus_loader = CorpusLoader(
        corpus_dir=str(corpus_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs_per_folder=max_docs_per_folder,
        logger=pipeline_logger,
    )
    qa_loader = KorMedMCQALoader(str(dataset_root), logger=pipeline_logger)

    chunks = corpus_loader.load_korean_corpus()
    retriever = BM25Retriever(logger=pipeline_logger, strict_kiwi=True)
    retriever.build_index(chunks)

    requested_splits = split_specs or KORMEDMCQA_SPLITS
    split_records = qa_loader.load_splits(requested_splits)

    manifest: Dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "dataset": "KorMedMCQA",
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "schema": "v1",
            "workspace_root": str(workspace_root),
            "dataset_root": str(dataset_root),
            "corpus_dir": str(corpus_dir),
        },
        "splits": {},
    }

    total_cached_queries = 0
    for split_name in tqdm(requested_splits, desc="Caching KorMedMCQA splits"):
        normalized_split = split_name.strip()
        if normalized_split == "doctor_trian":
            normalized_split = "doctor_train"

        records = split_records[normalized_split]
        entries: List[Dict[str, Any]] = []
        index_by_qhash: Dict[str, int] = {}

        for qa in records:
            question = qa.get("question", "")
            retrieval_query = build_retrieval_query(
                question=question,
                dataset="KorMedMCQA",
                options=qa.get("options"),
            )
            retrieved = retriever.search(retrieval_query, top_k=top_k)
            entry = build_cache_entry(normalized_split, qa, retrieved, retrieval_query)
            entries.append(entry)
            index_by_qhash[question_hash(retrieval_query)] = len(entries) - 1

        cache_data = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "dataset": "KorMedMCQA",
                "split": normalized_split,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(chunks),
                "total_queries": len(entries),
                "schema": "v1",
            },
            "index_by_qhash": index_by_qhash,
            "entries": entries,
        }

        output_file = output_root / f"KorMedMCQA_{normalized_split}.json"
        SparseRetrievalCache.save(cache_data, str(output_file))

        manifest["splits"][normalized_split] = {
            "file": output_file.name,
            "num_queries": len(entries),
        }
        total_cached_queries += len(entries)

        pipeline_logger.info(
            "Cached KorMedMCQA split=%s queries=%s file=%s",
            normalized_split,
            len(entries),
            output_file,
        )

    manifest["meta"]["total_splits"] = len(requested_splits)
    manifest["meta"]["total_cached_queries"] = total_cached_queries

    manifest_path = output_root / "KorMedMCQA_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    pipeline_logger.info("KorMedMCQA sparse cache build completed: %s", output_root)
    pipeline_logger.info("Manifest saved: %s", manifest_path)
    return manifest


def resolve_output_root(workspace_root: Path, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return workspace_root / "retrieval_cache" / "bm25"


def find_workspace_root(file_path: Path) -> Path:
    for parent in file_path.resolve().parents:
        if (parent / "data").exists() and (parent / "keys.py").exists():
            return parent
    raise RuntimeError(f"Unable to locate workspace root from {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 sparse cache for KorMedMCQA")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-docs-per-folder", type=int, default=None)
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=KORMEDMCQA_SPLITS,
        help="KorMedMCQA splits to cache (default: all six train/dev/test splits)",
    )
    args = parser.parse_args()

    workspace_root = find_workspace_root(Path(__file__))
    dataset_root = Path(args.dataset_root) if args.dataset_root else workspace_root / "KorMedMCQA"
    corpus_dir = workspace_root / "data" / "train" / "corpus"
    output_root = resolve_output_root(workspace_root, args.output_root)

    build_kormedmcqa_cache(
        workspace_root=workspace_root,
        dataset_root=dataset_root,
        corpus_dir=corpus_dir,
        output_root=output_root,
        top_k=args.top_k,
        chunk_size=1000,
        chunk_overlap=100,
        max_docs_per_folder=args.max_docs_per_folder,
        split_specs=args.splits,
    )