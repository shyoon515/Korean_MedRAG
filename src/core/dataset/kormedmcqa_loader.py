"""
KorMedMCQA parquet dataset loader.

Loads the dentist/doctor train, dev, and test splits and normalizes them into
the same question-centric structure used by the sparse retrieval cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None


class KorMedMCQALoader:
    """Loader for the KorMedMCQA parquet files."""

    def __init__(self, dataset_dir: str, logger: Optional[logging.Logger] = None):
        self.dataset_dir = Path(dataset_dir)
        self.logger = logger or logging.getLogger(__name__)

    def load_split(self, split_spec: str) -> List[Dict[str, Any]]:
        """Load one split such as dentist_train or doctor_test."""
        normalized_spec = self._normalize_split_spec(split_spec)
        config_name, split_name = normalized_spec.split("_", 1)
        parquet_path = self._find_split_file(config_name, split_name)

        if pq is None:
            raise ModuleNotFoundError(
                "pyarrow is required to read KorMedMCQA parquet files"
            ) from _PYARROW_IMPORT_ERROR

        table = pq.read_table(parquet_path)
        rows = table.to_pylist()

        records: List[Dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            question = row.get("question", "") or ""
            q_number = row.get("q_number", row_index)
            records.append(
                {
                    "dataset": "KorMedMCQA",
                    "config_name": config_name,
                    "split": split_name,
                    "source": f"{config_name}_{split_name}",
                    "row_index": row_index,
                    "question_id": f"{config_name}_{split_name}_{q_number}",
                    "subject": row.get("subject"),
                    "year": row.get("year"),
                    "period": row.get("period"),
                    "q_number": q_number,
                    "question": question,
                    "options": {
                        "A": row.get("A", "") or "",
                        "B": row.get("B", "") or "",
                        "C": row.get("C", "") or "",
                        "D": row.get("D", "") or "",
                        "E": row.get("E", "") or "",
                    },
                    "answer": row.get("answer"),
                    "cot": row.get("cot", "") or "",
                }
            )

        self.logger.info(
            "Loaded KorMedMCQA split=%s rows=%s file=%s",
            normalized_spec,
            len(records),
            parquet_path,
        )
        return records

    def load_splits(self, split_specs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Load multiple splits and return them keyed by normalized split spec."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for split_spec in split_specs:
            normalized_spec = self._normalize_split_spec(split_spec)
            grouped[normalized_spec] = self.load_split(normalized_spec)
        return grouped

    def _normalize_split_spec(self, split_spec: str) -> str:
        spec = split_spec.strip()
        if spec == "doctor_trian":
            spec = "doctor_train"
        if spec.count("_") != 1:
            raise ValueError(f"Invalid KorMedMCQA split spec: {split_spec}")
        config_name, split_name = spec.split("_", 1)
        if config_name not in {"dentist", "doctor"}:
            raise ValueError(f"Unsupported KorMedMCQA config: {config_name}")
        if split_name not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported KorMedMCQA split: {split_name}")
        return f"{config_name}_{split_name}"

    def _find_split_file(self, config_name: str, split_name: str) -> Path:
        split_dir = self.dataset_dir / config_name
        if not split_dir.exists():
            raise FileNotFoundError(f"KorMedMCQA folder not found: {split_dir}")

        candidates = sorted(split_dir.glob(f"{split_name}-*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"KorMedMCQA parquet not found for {config_name}_{split_name} under {split_dir}"
            )
        return candidates[0]