"""
Run sparse-only generation with OpenAI gpt-4o-mini.

Targets:
- KorMedMCQA: 2 queries per split (dentist/doctor x train/dev/test)
- data QA folders: 2 queries per source folder for train and test

Outputs:
- JSON result file under outputs/generation/
- Generation logs under outputs/generation/logs/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure workspace root is importable when this file is run directly.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.dataset.qa_loader import QALoader
from src.core.dataset.kormedmcqa_loader import KorMedMCQALoader
from src.core.chain.rag_chain import RAGChain
from src.core.utils.logging_utils import setup_category_loggers


KORMED_SPLITS = [
    "dentist_train",
    "dentist_dev",
    "dentist_test",
    "doctor_train",
    "doctor_dev",
    "doctor_test",
]


def _sample_data_queries(workspace_root: Path, samples_per_source: int) -> List[Dict[str, Any]]:
    qa_loader = QALoader(
        train_qa_dir=str(workspace_root / "data" / "train" / "qa"),
        test_qa_dir=str(workspace_root / "data" / "test" / "qa"),
    )

    sampled: List[Dict[str, Any]] = []
    for split, base_dir in (("train", qa_loader.train_qa_dir), ("test", qa_loader.test_qa_dir)):
        for folder in sorted([d for d in base_dir.iterdir() if d.is_dir()]):
            rows = qa_loader.load_qa_from_folder(folder)
            picks = rows[:samples_per_source]
            for idx, row in enumerate(picks):
                sampled.append(
                    {
                        "dataset": "data",
                        "group": "data",
                        "split": split,
                        "source": folder.name,
                        "sample_index": idx,
                        "qa_id": row.get("qa_id"),
                        "q_type": row.get("q_type"),
                        "question": row.get("question", ""),
                        "answer": row.get("answer"),
                    }
                )
    return sampled


def _sample_kormed_queries(workspace_root: Path, samples_per_source: int) -> List[Dict[str, Any]]:
    loader = KorMedMCQALoader(str(workspace_root / "KorMedMCQA"))
    sampled: List[Dict[str, Any]] = []

    for split in KORMED_SPLITS:
        rows = loader.load_split(split)
        picks = rows[:samples_per_source]
        for idx, row in enumerate(picks):
            sampled.append(
                {
                    "dataset": "KorMedMCQA",
                    "group": "KorMedMCQA",
                    "split": split,
                    "source": split,
                    "sample_index": idx,
                    "q_type": 1,
                    "question": row.get("question", ""),
                    "options": row.get("options", {}),
                    "answer": row.get("answer"),
                    "question_id": row.get("question_id"),
                }
            )
    return sampled


def _run_generation(
    chain: RAGChain,
    rows: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    outputs = chain.ask(rows)

    results: List[Dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        generated_text, retrieved = output
        is_kormed = str(row.get("dataset", "")).lower() == "kormedmcqa"
        prompt_profile = "kormed_mcq_1to5" if is_kormed else f"data_qtype_{row.get('q_type', 'unknown')}"
        results.append(
            {
                **row,
                "prompt_profile": prompt_profile,
                "generated": generated_text,
                "retrieved_count": len(retrieved),
                "retrieval_hit": len(retrieved) > 0,
                "retrieved_top_k": retrieved[:top_k],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-only RAG generation runner")
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cache-root", type=str, default="retrieval_cache/bm25")
    parser.add_argument("--output-dir", type=str, default="outputs/generation")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    cache_root = (workspace_root / args.cache_root).resolve()
    output_dir = (workspace_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loggers = setup_category_loggers(
        log_dir=str(output_dir / "logs"),
        base_name="sparse_generation",
    )
    pipeline_logger = loggers["pipeline"]
    generation_logger = loggers["generation"]

    pipeline_logger.info("Sparse generation run started")
    pipeline_logger.info("cache_root=%s output_dir=%s", cache_root, output_dir)

    # Data QA queries use by-source manifest in retrieval_cache/bm25/manifest.json.
    data_chain = RAGChain(
        retrieval_mode="sparse_only",
        top_k=args.top_k,
        generator_type="openai",
        generator_name="gpt-4o-mini",
        sparse_cache_path=str(cache_root),
        include_retrieval_results=True,
        logger=generation_logger,
    )

    data_rows = _sample_data_queries(workspace_root, args.samples_per_source)
    pipeline_logger.info("Sampled data queries: %s", len(data_rows))
    data_results = _run_generation(data_chain, data_rows, args.top_k)

    # KorMedMCQA splits may have separate cache files.
    kormed_rows = _sample_kormed_queries(workspace_root, args.samples_per_source)
    pipeline_logger.info("Sampled KorMedMCQA queries: %s", len(kormed_rows))

    kormed_results: List[Dict[str, Any]] = []
    split_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in kormed_rows:
        split_to_rows.setdefault(row["split"], []).append(row)

    for split, rows in split_to_rows.items():
        split_cache_file = cache_root / f"KorMedMCQA_{split}.json"
        if split_cache_file.exists():
            sparse_cache_path = str(split_cache_file)
            pipeline_logger.info("Using KorMed cache for split=%s file=%s", split, split_cache_file)
        else:
            # Fallback to common cache root; retrieval may miss if qhash is absent.
            sparse_cache_path = str(cache_root)
            pipeline_logger.warning(
                "KorMed cache file missing for split=%s, fallback cache root will be used", split
            )

        split_chain = RAGChain(
            retrieval_mode="sparse_only",
            top_k=args.top_k,
            generator_type="openai",
            generator_name="gpt-4o-mini",
            sparse_cache_path=sparse_cache_path,
            include_retrieval_results=True,
            logger=generation_logger,
        )
        kormed_results.extend(_run_generation(split_chain, rows, args.top_k))

    all_results = data_results + kormed_results
    total_hits = sum(1 for r in all_results if r["retrieval_hit"])

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "retrieval_mode": "sparse_only",
            "samples_per_source": args.samples_per_source,
            "top_k": args.top_k,
            "cache_root": str(cache_root),
            "workspace_root": str(workspace_root),
            "total_generations": len(all_results),
            "retrieval_hit_count": total_hits,
            "retrieval_miss_count": len(all_results) - total_hits,
        },
        "results": all_results,
    }

    output_file = output_dir / f"sparse_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    pipeline_logger.info("Sparse generation run completed: %s", output_file)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
