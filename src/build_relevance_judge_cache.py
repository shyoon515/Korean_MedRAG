"""
Build LLM-as-judge relevance cache for retrieval results.

- Evaluates retrieval relevance for top-1 to top-N (default: 10) per query.
- Uses binary relevance labels: 0 (not relevant) or 1 (relevant).
- Writes annotated cache files to a separate output root.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure workspace root is importable when this file is run directly.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.generator.llm import OpenAIGenerator
from src.core.utils.logging_utils import setup_category_loggers


class RetrievalRelevanceJudge:
    """Binary relevance judge using an LLM."""

    def __init__(self, generator: OpenAIGenerator):
        self.generator = generator

    def judge(self, question: str, context: str) -> Dict[str, Any]:
        prompt = self._build_prompt(question=question, context=context)
        raw = self.generator.generate([prompt])[0]
        
        ####
        try:
            result = int(raw.strip())
            relevance = 1 if result == 1 else 0
            return {        "relevance": relevance,     "reason": "dummy_reason", "raw": raw}
        except:
            return {        "relevance": 0, "reason": "parse_error", "raw": raw}
        ####

        return self._parse(raw)

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "당신은 의료 검색 결과의 관련성을 평가하는 심사자입니다.\n"
            "질문과 문맥을 보고, 문맥이 질문에 답하는 데 실질적으로 관련 있으면 1, 아니면 0을 부여하세요.\n"
            "출력은 반드시 0 또는 1만 출력하세요.\n"
            "형식: 0 또는 1\n\n"
            f"[질문]\n{question}\n\n"
            f"[문맥]\n{context}\n"
        )

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                value = int(parsed.get("relevance", 0))
                value = 1 if value == 1 else 0
                reason = str(parsed.get("reason", "")).strip()
                return {
                    "relevance": value,
                    "reason": reason,
                    "raw": raw,
                }
            except Exception:
                pass

        fallback_value = 1 if re.search(r"\b1\b", raw) else 0
        return {
            "relevance": fallback_value,
            "reason": "parse_fallback",
            "raw": raw,
        }


def _resolve_cache_files(cache_root: Path) -> List[Path]:
    if cache_root.is_file() and cache_root.name == "manifest.json":
        return _resolve_manifest_files(cache_root)

    if cache_root.is_file():
        return [cache_root]

    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache path not found: {cache_root}")

    manifest_path = cache_root / "manifest.json"
    files: List[Path] = []
    if manifest_path.exists():
        files.extend(_resolve_manifest_files(manifest_path))

    # Include explicit split cache files (e.g., KorMedMCQA_doctor_train.json)
    for p in sorted(cache_root.glob("*.json")):
        if p.name == "manifest.json":
            continue
        if p not in files:
            files.append(p)

    return files


def _resolve_manifest_files(manifest_path: Path) -> List[Path]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_dir = manifest_path.parent
    files: List[Path] = []
    for source in manifest.get("sources", {}).values():
        name = source.get("file")
        if not name:
            continue
        fpath = base_dir / name
        if fpath.exists() and fpath.suffix.lower() == ".json":
            files.append(fpath)
    return files


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _upsert_meta(data: Dict[str, Any], model: str, max_rank: int) -> None:
    meta = data.setdefault("meta", {})
    judge_meta = meta.setdefault("llm_relevance_judge", {})
    judge_meta["model"] = model
    judge_meta["max_rank"] = max_rank
    judge_meta["updated_at"] = datetime.now().isoformat()


def _evaluate_file(
    src_path: Path,
    dst_path: Path,
    judge: RetrievalRelevanceJudge,
    model: str,
    max_rank: int,
    overwrite: bool,
    max_entries: Optional[int],
    logger,
) -> Tuple[int, int]:
    src_data = _load_json(src_path)

    if dst_path.exists() and not overwrite:
        out_data = _load_json(dst_path)
    else:
        out_data = copy.deepcopy(src_data)

    entries: List[Dict[str, Any]] = out_data.get("entries", [])
    if max_entries is not None:
        entries = entries[:max_entries]

    judged_items = 0
    positive_items = 0

    for entry_idx, entry in enumerate(entries):
        question = str(entry.get("question", "")).strip()
        if not question:
            continue

        retrieved_items = entry.get("retrieved_items", [])
        if not isinstance(retrieved_items, list):
            continue

        local_positive = 0
        local_judged = 0
        for rank_idx, item in enumerate(retrieved_items[:max_rank], start=1):
            if (not overwrite) and ("llm_relevance_01" in item):
                local_judged += 1
                if int(item.get("llm_relevance_01", 0)) == 1:
                    local_positive += 1
                continue

            context = str(item.get("text", "")).strip()
            if not context:
                item["llm_relevance_01"] = 0
                item["llm_relevance_reason"] = "empty_context"
                item["llm_relevance_model"] = model
                local_judged += 1
                continue

            judged = judge.judge(question=question, context=context)
            relevance = int(judged.get("relevance", 0))
            relevance = 1 if relevance == 1 else 0

            item["llm_relevance_01"] = relevance
            item["llm_relevance_reason"] = judged.get("reason", "")
            item["llm_relevance_model"] = model
            item["llm_relevance_rank_scope"] = rank_idx

            local_judged += 1
            if relevance == 1:
                local_positive += 1

        evaluated_items = retrieved_items[:max_rank]
        labels_list: List[int] = [int(item.get("llm_relevance_01", 0)) for item in evaluated_items]
        labels_by_rank: Dict[str, int] = {
            str(idx): label for idx, label in enumerate(labels_list, start=1)
        }
        labels_by_chunk_id: Dict[str, int] = {}
        for item, label in zip(evaluated_items, labels_list):
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id:
                labels_by_chunk_id[chunk_id] = label

        entry["llm_relevance_summary"] = {
            "evaluated_top_k": min(max_rank, len(retrieved_items)),
            "relevant_count": local_positive,
            "irrelevant_count": max(local_judged - local_positive, 0),
            "relevant_ratio": (local_positive / local_judged) if local_judged > 0 else 0.0,
        }
        entry["llm_relevance_labels"] = labels_list
        entry["llm_relevance_labels_by_rank"] = labels_by_rank
        entry["llm_relevance_labels_by_chunk_id"] = labels_by_chunk_id

        judged_items += local_judged
        positive_items += local_positive

        if logger and (entry_idx + 1) % 100 == 0:
            logger.info("Progress %s: %s/%s entries processed", src_path.name, entry_idx + 1, len(entries))

    _upsert_meta(out_data, model=model, max_rank=max_rank)
    _save_json(dst_path, out_data)

    return judged_items, positive_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM relevance-judge cache for retrieval results")
    parser.add_argument("--cache-root", type=str, default="retrieval_cache/bm25")
    parser.add_argument("--output-root", type=str, default="retrieval_cache/bm25_llmjudge")
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--max-entries-per-file", type=int, default=0, help="0 means all entries")
    parser.add_argument(
        "--include-files",
        type=str,
        default="",
        help="Comma-separated cache filenames to evaluate (e.g. TL_기타.json,VL_기타.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_rank < 1:
        raise ValueError("--max-rank must be >= 1")

    workspace_root = Path(__file__).resolve().parents[1]
    cache_root = (workspace_root / args.cache_root).resolve()
    output_root = (workspace_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    loggers = setup_category_loggers(
        log_dir=str(output_root / "logs"),
        base_name="relevance_judge_cache",
    )
    pipeline_logger = loggers["pipeline"]

    pipeline_logger.info("Relevance judge cache build started")
    pipeline_logger.info("cache_root=%s output_root=%s model=%s", cache_root, output_root, args.model)

    cache_files = _resolve_cache_files(cache_root)
    include_files = [name.strip() for name in args.include_files.split(",") if name.strip()]
    if include_files:
        include_set = set(include_files)
        cache_files = [p for p in cache_files if p.name in include_set]
        missing = sorted(include_set - {p.name for p in cache_files})
        if missing:
            pipeline_logger.warning("Some --include-files were not found under cache root: %s", missing)

    if args.max_files > 0:
        cache_files = cache_files[: args.max_files]

    if not cache_files:
        raise ValueError("No cache files selected for evaluation. Check --cache-root/--include-files")

    generator = OpenAIGenerator(model_name=args.model, logger=loggers["generation"])
    judge = RetrievalRelevanceJudge(generator=generator)

    total_judged = 0
    total_positive = 0
    max_entries = args.max_entries_per_file if args.max_entries_per_file > 0 else None

    for src_path in cache_files:
        dst_path = output_root / src_path.name
        pipeline_logger.info("Evaluating file: %s -> %s", src_path.name, dst_path.name)

        judged_count, positive_count = _evaluate_file(
            src_path=src_path,
            dst_path=dst_path,
            judge=judge,
            model=args.model,
            max_rank=args.max_rank,
            overwrite=args.overwrite,
            max_entries=max_entries,
            logger=pipeline_logger,
        )

        total_judged += judged_count
        total_positive += positive_count

        ratio = (positive_count / judged_count) if judged_count > 0 else 0.0
        pipeline_logger.info(
            "Completed file=%s judged=%s relevant=%s relevant_ratio=%.4f",
            src_path.name,
            judged_count,
            positive_count,
            ratio,
        )

    total_ratio = (total_positive / total_judged) if total_judged > 0 else 0.0
    pipeline_logger.info(
        "Relevance judge cache build completed files=%s judged=%s relevant=%s relevant_ratio=%.4f",
        len(cache_files),
        total_judged,
        total_positive,
        total_ratio,
    )

    print(f"Completed. files={len(cache_files)} judged={total_judged} relevant={total_positive}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
