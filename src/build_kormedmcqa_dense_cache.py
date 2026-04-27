"""
Build dense FAISS index for KorMedMCQA experiments.

This script intentionally uses the same corpus preprocessing as sparse caches:
- corpus source: data/train/corpus/TS_국문*
- chunking: chunk_size=1000, chunk_overlap=100
- encoder: BAAI/bge-m3

A dedicated output directory is used so KorMedMCQA experiments can manage their
own dense index artifacts independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _load_core_components():
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    try:
        from src.core import (
            CorpusLoader,
            DenseRetriever,
            KorMedMCQALoader,
            SparseRetrievalCache,
            question_hash,
            setup_category_loggers,
        )
    except ImportError:
        from core import (
            CorpusLoader,
            DenseRetriever,
            KorMedMCQALoader,
            SparseRetrievalCache,
            question_hash,
            setup_category_loggers,
        )

    return (
        CorpusLoader,
        DenseRetriever,
        KorMedMCQALoader,
        SparseRetrievalCache,
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


def build_cache_entry(split_name: str, qa: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dataset": "KorMedMCQA",
        "split": split_name,
        "source": split_name,
        "question_id": qa.get("question_id"),
        "question": qa.get("question", ""),
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


def build_kormedmcqa_dense_index(
    workspace_root: Path,
    dataset_root: Path,
    corpus_dir: Path,
    output_root: Path,
    index_root: Path,
    model_name: str = "BAAI/bge-m3",
    top_k: int = 20,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    max_docs_per_folder: int | None = None,
    split_specs: List[str] | None = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    (
        CorpusLoader,
        DenseRetriever,
        KorMedMCQALoader,
        SparseRetrievalCache,
        question_hash,
        setup_category_loggers,
    ) = _load_core_components()

    output_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    log_dir = output_root / "logs"
    loggers = setup_category_loggers(
        str(log_dir),
        level=logging.INFO,
        base_name="kormedmcqa_dense_index",
    )
    pipeline_logger = loggers["pipeline"]

    pipeline_logger.info("KorMedMCQA dense index build started")
    pipeline_logger.info(
        "Config: model=%s top_k=%s chunk_size=%s chunk_overlap=%s max_docs_per_folder=%s batch_size=%s dataset_root=%s corpus_dir=%s output_root=%s index_root=%s",
        model_name,
        top_k,
        chunk_size,
        chunk_overlap,
        max_docs_per_folder,
        batch_size,
        dataset_root,
        corpus_dir,
        output_root,
        index_root,
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"KorMedMCQA dataset root not found: {dataset_root}")

    corpus_loader = CorpusLoader(
        corpus_dir=str(corpus_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs_per_folder=max_docs_per_folder,
        logger=pipeline_logger,
    )
    qa_loader = KorMedMCQALoader(str(dataset_root), logger=pipeline_logger)

    chunks = corpus_loader.load_korean_corpus()

    retriever = DenseRetriever(
        model_name=model_name,
        batch_size=batch_size,
        logger=pipeline_logger,
    )
    retriever.build_index(chunks)

    index_dir = index_root / "kormedmcqa_corpus_index"
    retriever.save_index(str(index_dir))

    requested_splits = split_specs or KORMEDMCQA_SPLITS
    split_records = qa_loader.load_splits(requested_splits)

    manifest = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "dataset": "KorMedMCQA",
            "index_type": "faiss_dense",
            "encoder": model_name,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "schema": "v1",
            "workspace_root": str(workspace_root),
            "dataset_root": str(dataset_root),
            "corpus_dir": str(corpus_dir),
        },
        "index": {
            "index_dir": str(index_dir),
            "index_file": "index.faiss",
            "chunks_file": "chunks.jsonl",
            "meta_file": "meta.json",
        },
        "splits": {},
    }

    total_cached_queries = 0
    for split_name in requested_splits:
        normalized_split = split_name.strip()
        if normalized_split == "doctor_trian":
            normalized_split = "doctor_train"

        records = split_records[normalized_split]
        entries: List[Dict[str, Any]] = []
        index_by_qhash: Dict[str, int] = {}

        for qa in records:
            question = qa.get("question", "")
            retrieved = retriever.search(question, top_k=top_k)
            entry = build_cache_entry(normalized_split, qa, retrieved)
            entries.append(entry)
            index_by_qhash[question_hash(question)] = len(entries) - 1

        cache_data = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "dataset": "KorMedMCQA",
                "retriever": "dense",
                "encoder": model_name,
                "split": normalized_split,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(chunks),
                "total_queries": len(entries),
                "index_dir": str(index_dir),
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
            "Cached KorMedMCQA dense split=%s queries=%s file=%s",
            normalized_split,
            len(entries),
            output_file,
        )

    manifest["meta"]["total_splits"] = len(requested_splits)
    manifest["meta"]["total_cached_queries"] = total_cached_queries

    manifest_path = output_root / "KorMedMCQA_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    pipeline_logger.info("KorMedMCQA dense index build completed: %s", index_dir)
    pipeline_logger.info("Manifest saved: %s", manifest_path)
    return manifest


def resolve_output_root(workspace_root: Path, output_root: str | None) -> Path:
    if output_root is not None:
        return Path(output_root)
    return workspace_root / "retrieval_cache" / "dense"


def resolve_index_root(workspace_root: Path, index_root: str | None) -> Path:
    if index_root is not None:
        return Path(index_root)
    return workspace_root / "dense_index" / "kormedmcqa"


def find_workspace_root(file_path: Path) -> Path:
    workspace_root = file_path.resolve().parents[1]
    if not (workspace_root / "data").exists():
        raise RuntimeError(f"Workspace data directory not found under {workspace_root}")
    return workspace_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dense FAISS index for KorMedMCQA")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--index-root", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="BAAI/bge-m3")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
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
    index_root = resolve_index_root(workspace_root, args.index_root)

    build_kormedmcqa_dense_index(
        workspace_root=workspace_root,
        dataset_root=dataset_root,
        corpus_dir=corpus_dir,
        output_root=output_root,
        index_root=index_root,
        model_name=args.encoder,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_docs_per_folder=args.max_docs_per_folder,
        split_specs=args.splits,
        batch_size=args.batch_size,
    )
