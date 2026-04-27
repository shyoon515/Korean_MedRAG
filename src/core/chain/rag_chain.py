import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..generator import OpenAIGenerator, PromptGenerator, VLLMGenerator
from ..utils import build_retrieval_query, question_hash


@dataclass
class _CacheBundle:
    by_qhash: Dict[str, Dict[str, Any]]


class RAGChain:
    """Cache-based RAG chain with pluggable generators and retrieval fusion modes."""

    SUPPORTED_MODES = {
        "llm_only",
        "sparse_only",
        "dense_only",
        "reciprocal_rank",
        "alpha",
    }

    def __init__(
        self,
        retrieval_mode: str = "sparse_only",
        top_k: int = 5,
        fusion_top_n: int = 10,
        alpha: float = 0.5,
        seq_type: str = "dq",
        generator_type: str = "openai",
        generator_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        vllm_api_base: str = "http://localhost:8000/v1",
        sparse_cache_path: Optional[str] = None,
        dense_cache_path: Optional[str] = None,
        include_retrieval_results: bool = True,
        rrf_k: int = 60,
        logger=None,
    ):
        self.retrieval_mode = self._normalize_mode(retrieval_mode)
        self.top_k = top_k
        self.fusion_top_n = fusion_top_n
        self.alpha = alpha
        self.seq_type = seq_type
        self.include_retrieval_results = include_retrieval_results
        self.rrf_k = rrf_k
        self.logger = logger

        self.sparse_cache = self._load_cache_bundle(sparse_cache_path) if sparse_cache_path else None
        self.dense_cache = self._load_cache_bundle(dense_cache_path) if dense_cache_path else None

        # Placeholder for compatibility with legacy initialization sequence.
        self.retriever = self._load_retriever_dummy()

        self.generator = self._load_generator(
            generator_name=generator_name,
            generator_type=generator_type,
            openai_api_key=openai_api_key,
            vllm_api_base=vllm_api_base,
        )

    def ask(
        self,
        question: Union[str, Dict[str, Any], List[str], List[Dict[str, Any]]],
    ) -> Union[List[str], List[Tuple[str, List[Dict[str, Any]]]]]:
        rows: List[Dict[str, Any]]
        if isinstance(question, str):
            rows = [{"question": question}]
        elif isinstance(question, dict):
            rows = [question]
        elif question and isinstance(question[0], dict):
            rows = question  # type: ignore[assignment]
        else:
            rows = [{"question": q} for q in question]  # type: ignore[arg-type]

        prompts: List[str] = []
        retrieved_all: List[List[Dict[str, Any]]] = []

        for row in rows:
            q = row.get("question", "")
            retrieval_query = build_retrieval_query(
                question=q,
                dataset=row.get("dataset"),
                options=row.get("options"),
            )
            retrieved = self.retrieve(retrieval_query)
            if not retrieved and retrieval_query != q:
                # Backward compatibility with older caches keyed by raw question.
                retrieved = self.retrieve(q)
            retrieved_all.append(retrieved)

            if self.retrieval_mode == "llm_only":
                prompt = PromptGenerator.generate_answer_without_docs(
                    q,
                    dataset=row.get("dataset"),
                    q_type=row.get("q_type"),
                    options=row.get("options"),
                    metadata=row,
                )[0]
            else:
                docs = [item.get("text", "") for item in retrieved]
                prompt = PromptGenerator.generate_answer_with_docs(
                    docs=docs,
                    question=q,
                    seq_type=self.seq_type,
                    dataset=row.get("dataset"),
                    q_type=row.get("q_type"),
                    options=row.get("options"),
                    metadata=row,
                )[0]
            prompts.append(prompt)

        outputs = self.generator.generate(prompts)

        if self.include_retrieval_results:
            return list(zip(outputs, retrieved_all))
        return outputs

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        if self.retrieval_mode == "llm_only":
            return []

        if self.retrieval_mode == "sparse_only":
            sparse = self._lookup(self.sparse_cache, question)
            return sparse[: self.top_k]

        if self.retrieval_mode == "dense_only":
            dense = self._lookup(self.dense_cache, question)
            return dense[: self.top_k]

        sparse = self._lookup(self.sparse_cache, question)
        dense = self._lookup(self.dense_cache, question)

        if self.retrieval_mode == "reciprocal_rank":
            return self._fuse_rrf(sparse, dense, top_k=self.top_k, k_rrf=self.rrf_k)

        if self.retrieval_mode == "alpha":
            return self._fuse_alpha(
                sparse,
                dense,
                top_k=self.top_k,
                top_n=self.fusion_top_n,
                alpha=self.alpha,
            )

        raise ValueError(f"Unsupported retrieval mode: {self.retrieval_mode}")

    def _load_retriever_dummy(self):
        if self.logger:
            self.logger.info("[RAGChain] retriever loading skipped (cache-only mode)")
        return None

    def _load_generator(
        self,
        generator_name: str,
        generator_type: str,
        openai_api_key: Optional[str],
        vllm_api_base: str,
    ):
        if generator_type == "openai":
            return OpenAIGenerator(model_name=generator_name, api_key=openai_api_key, logger=self.logger)
        if generator_type == "vllm":
            return VLLMGenerator(model_name=generator_name, api_base=vllm_api_base, logger=self.logger)
        raise ValueError("generator_type must be one of ['openai', 'vllm']")

    def _lookup(self, cache_bundle: Optional[_CacheBundle], question: str) -> List[Dict[str, Any]]:
        if cache_bundle is None:
            raise RuntimeError("Required retrieval cache is not loaded.")

        qh = question_hash(question)
        entry = cache_bundle.by_qhash.get(qh)
        if not entry:
            if self.logger:
                self.logger.warning("[RAGChain] question not found in cache: %s", question[:80])
            return []

        retrieved = entry.get("retrieved_items", [])
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(retrieved, start=1):
            normalized.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0.0)),
                    "rank": int(item.get("rank", idx)),
                    "doc_id": item.get("doc_id"),
                    "source": item.get("source"),
                }
            )
        return normalized

    @classmethod
    def _normalize_mode(cls, mode: str) -> str:
        key = mode.strip().lower().replace("-", "_").replace(" ", "_")
        alias = {
            "llm": "llm_only",
            "sparse": "sparse_only",
            "dense": "dense_only",
            "rrf": "reciprocal_rank",
            "reciprocalrank": "reciprocal_rank",
            "weighted": "alpha",
            "alpha_weighted": "alpha",
        }
        key = alias.get(key, key)
        if key not in cls.SUPPORTED_MODES:
            raise ValueError(f"retrieval_mode must be one of {sorted(cls.SUPPORTED_MODES)}")
        return key

    @staticmethod
    def _load_cache_bundle(path_value: str) -> _CacheBundle:
        root = Path(path_value)
        files = []

        if root.is_file() and root.name == "manifest.json":
            files = RAGChain._resolve_manifest_files(root)
        elif root.is_file():
            files = [root]
        elif root.is_dir():
            manifest = root / "manifest.json"
            if manifest.exists():
                files = RAGChain._resolve_manifest_files(manifest)
            else:
                files = [p for p in root.glob("*.json") if p.name != "manifest.json"]
        else:
            raise FileNotFoundError(f"Cache path does not exist: {path_value}")

        by_qhash: Dict[str, Dict[str, Any]] = {}
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            idx_map = data.get("index_by_qhash", {})
            entries = data.get("entries", [])
            for qh, idx in idx_map.items():
                if qh in by_qhash:
                    continue
                if 0 <= idx < len(entries):
                    by_qhash[qh] = entries[idx]

        return _CacheBundle(by_qhash=by_qhash)

    @staticmethod
    def _resolve_manifest_files(manifest_path: Path) -> List[Path]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        base_dir = manifest_path.parent
        sources = manifest.get("sources", {})
        files: List[Path] = []
        for source in sources.values():
            name = source.get("file")
            if not name:
                continue
            fpath = base_dir / name
            if fpath.exists() and fpath.suffix.lower() == ".json":
                files.append(fpath)
        return files

    @staticmethod
    def _fuse_rrf(
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int,
        k_rrf: int = 60,
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[str, float] = {}
        item_info: Dict[str, Dict[str, Any]] = {}

        for result in sparse_results:
            cid = str(result.get("chunk_id"))
            rank = int(result.get("rank", 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            item_info.setdefault(cid, result)

        for result in dense_results:
            cid = str(result.get("chunk_id"))
            rank = int(result.get("rank", 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            item_info.setdefault(cid, result)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out: List[Dict[str, Any]] = []
        for idx, (cid, score) in enumerate(ranked, start=1):
            base = dict(item_info[cid])
            base["rank"] = idx
            base["rrf_score"] = float(score)
            base["score"] = float(score)
            base["method"] = "reciprocal_rank"
            out.append(base)
        return out

    @staticmethod
    def _fuse_alpha(
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int,
        top_n: int,
        alpha: float,
    ) -> List[Dict[str, Any]]:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")

        sparse_top = sparse_results[:top_n]
        dense_top = dense_results[:top_n]

        sparse_norm = RAGChain._minmax_score_map(sparse_top)
        dense_norm = RAGChain._minmax_score_map(dense_top)

        item_info: Dict[str, Dict[str, Any]] = {}
        for item in sparse_top + dense_top:
            cid = str(item.get("chunk_id"))
            item_info.setdefault(cid, item)

        combined: Dict[str, float] = {}
        for cid in item_info.keys():
            s = sparse_norm.get(cid, 0.0)
            d = dense_norm.get(cid, 0.0)
            combined[cid] = (1.0 - alpha) * s + alpha * d

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out: List[Dict[str, Any]] = []
        for idx, (cid, score) in enumerate(ranked, start=1):
            base = dict(item_info[cid])
            base["rank"] = idx
            base["score"] = float(score)
            base["combined_score"] = float(score)
            base["sparse_score_scaled"] = float(sparse_norm.get(cid, 0.0))
            base["dense_score_scaled"] = float(dense_norm.get(cid, 0.0))
            base["alpha"] = alpha
            base["method"] = "alpha"
            out.append(base)
        return out

    @staticmethod
    def _minmax_score_map(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        rows = list(results)
        if not rows:
            return {}

        scores = [float(r.get("score", 0.0)) for r in rows]
        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return {str(r.get("chunk_id")): 1.0 for r in rows}

        denom = max_score - min_score
        out: Dict[str, float] = {}
        for r, raw in zip(rows, scores):
            out[str(r.get("chunk_id"))] = (raw - min_score) / denom
        return out
