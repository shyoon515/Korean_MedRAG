"""
전체 corpus를 인덱싱하고, query source별로 BM25 top-k 결과를 캐시로 저장.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    from src.core import (
        CorpusLoader,
        QALoader,
        BM25Retriever,
        setup_category_loggers,
        SparseRetrievalCache,
        question_hash,
    )
except ImportError:
    from core import (
        CorpusLoader,
        QALoader,
        BM25Retriever,
        setup_category_loggers,
        SparseRetrievalCache,
        question_hash,
    )


def collect_queries_by_source(
    qa_loader: QALoader,
    qa_train_dir: str,
    qa_test_dir: str,
) -> Dict[str, List[Tuple[str, Dict]]]:
    """QA를 source 폴더 단위로 수집한다. key 예: TL_기타, VL_기타"""
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


def sample_queries(
    rows: List[Tuple[str, Dict]],
    queries_per_source: int,
    seed: int,
) -> List[Tuple[str, Dict]]:
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


def build_cache(
    corpus_dir: str,
    qa_train_dir: str,
    qa_test_dir: str,
    output_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    top_k: int = 20,
    max_docs_per_folder: int = None,
    max_queries: int = None,
    output_root: str = None,
    queries_per_source: int = 10,
    sample_seed: int = 42,
):
    if output_root is None:
        raise ValueError("output_root must be provided")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    log_dir = output_root_path / "logs"
    loggers = setup_category_loggers(str(log_dir), level=logging.INFO, base_name="sparse_cache")
    pipeline_logger = loggers["pipeline"]

    pipeline_logger.info("Sparse cache build started")
    pipeline_logger.info(
        "Config: top_k=%s chunk_size=%s chunk_overlap=%s max_docs_per_folder=%s max_queries=%s queries_per_source=%s output_root=%s",
        top_k,
        chunk_size,
        chunk_overlap,
        max_docs_per_folder,
        max_queries,
        queries_per_source,
        output_root,
    )

    corpus_loader = CorpusLoader(
        corpus_dir=corpus_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs_per_folder=max_docs_per_folder,
        logger=pipeline_logger,
    )
    qa_loader = QALoader(qa_train_dir, qa_test_dir, logger=pipeline_logger)

    # 1) Full corpus load + index build
    chunks = corpus_loader.load_korean_corpus()
    retriever = BM25Retriever(logger=pipeline_logger, strict_kiwi=True)
    retriever.build_index(chunks)

    # 2) Load queries grouped by source folder
    by_source = collect_queries_by_source(qa_loader, qa_train_dir, qa_test_dir)
    source_names = sorted(by_source.keys())
    pipeline_logger.info("Total query sources discovered: %s", len(source_names))

    manifest = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "queries_per_source": queries_per_source,
            "sample_seed": sample_seed,
            "schema": "v1-by-source",
        },
        "sources": {},
    }

    total_cached_queries = 0
    source_iter = tqdm(source_names, desc="Caching query sources")
    for source_name in source_iter:
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
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": len(chunks),
                "total_queries": len(entries),
                "query_source": source_name,
                "schema": "v1",
            },
            "index_by_qhash": index_by_qhash,
            "entries": entries,
        }

        source_output = output_root_path / f"{source_name}.json"
        SparseRetrievalCache.save(cache_data, str(source_output))
        total_cached_queries += len(entries)

        manifest["sources"][source_name] = {
            "file": source_output.name,
            "num_cached_queries": len(entries),
            "total_available_queries": len(rows),
        }
        pipeline_logger.info(
            "Source cached: %s cached=%s total_available=%s file=%s",
            source_name,
            len(entries),
            len(rows),
            source_output,
        )

    manifest["meta"]["total_sources"] = len(source_names)
    manifest["meta"]["total_cached_queries"] = total_cached_queries

    manifest_path = output_root_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    pipeline_logger.info("Sparse cache build completed (by source): %s", output_root_path)
    pipeline_logger.info("Manifest saved: %s", manifest_path)
    return manifest


def resolve_output_root(workspace_root: Path, output_root: Optional[str]) -> str:
    if output_root is not None:
        return output_root

    # workspace root와 같은 층위에 retrieval_cache/bm25 생성
    default_root = workspace_root / "retrieval_cache" / "bm25"
    default_root.mkdir(parents=True, exist_ok=True)
    return str(default_root)


def find_workspace_root(file_path: Path) -> Path:
    for parent in file_path.resolve().parents:
        if (parent / "data").exists() and (parent / "keys.py").exists():
            return parent
    raise RuntimeError(f"Unable to locate workspace root from {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build sparse BM25 retrieval cache by query source")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--queries-per-source", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-docs-per-folder", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    workspace_root = find_workspace_root(Path(__file__))
    corpus_dir = workspace_root / "data" / "train" / "corpus"
    qa_train_dir = workspace_root / "data" / "train" / "qa"
    qa_test_dir = workspace_root / "data" / "test" / "qa"

    output_root = resolve_output_root(workspace_root, args.output_root)

    build_cache(
        corpus_dir=str(corpus_dir),
        qa_train_dir=str(qa_train_dir),
        qa_test_dir=str(qa_test_dir),
        output_path="",
        chunk_size=1000,
        chunk_overlap=100,
        top_k=args.top_k,
        max_docs_per_folder=args.max_docs_per_folder,
        max_queries=args.max_queries,
        output_root=output_root,
        queries_per_source=args.queries_per_source,
        sample_seed=args.sample_seed,
    )
