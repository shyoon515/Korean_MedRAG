import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path
from threading import Lock
from typing import List, Optional

from openai import OpenAI


def _load_root_openai_api_key() -> Optional[str]:
    """Load OPENAI_API_KEY from nearest ancestor keys.py if present."""
    # First, try regular import path.
    try:
        from keys import OPENAI_API_KEY as imported_key

        if imported_key:
            return imported_key
    except Exception:
        pass

    # Fallback: locate keys.py by walking up from this file.
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "keys.py"
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("project_keys", str(candidate))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            key = getattr(module, "OPENAI_API_KEY", None)
            if key:
                return key
        except Exception:
            continue
    return None


ROOT_OPENAI_API_KEY = _load_root_openai_api_key()


class OpenAIGenerator:
    """Text generator backed by the OpenAI Responses API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay_sec: float = 2.0,
        logger=None,
    ):
        resolved_api_key = api_key or ROOT_OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in root keys.py or environment.")

        self.client = OpenAI(api_key=resolved_api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.logger = logger

    def generate(self, prompts: List[str]) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        responses: List[str] = []
        for idx, prompt in enumerate(prompts):
            result = ""
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt,
                    )
                    result = (response.output_text or "").strip()
                    break
                except Exception as exc:  # pragma: no cover - network path
                    if self.logger:
                        self.logger.warning(
                            "[OpenAIGenerator] failed (%s/%s) for prompt %s: %s",
                            attempt,
                            self.max_retries,
                            idx + 1,
                            exc,
                        )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay_sec)

            responses.append(result)
            if self.logger:
                self.logger.info(
                    "[OpenAIGenerator] generated %s/%s\n[INPUT PROMPT]\n%s\n[OUTPUT]\n%s",
                    idx + 1,
                    len(prompts),
                    prompt,
                    result,
                )

        return responses


class VLLMGenerator:
    """Text generator that calls a vLLM OpenAI-compatible endpoint."""

    MODEL_ALIAS = {
        "exaone": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "midm": "K-intelligence/Midm-2.0-Mini-Instruct",
        "hyperclovax": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
    }

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_workers: int = 8,
        max_retries: int = 2,
        retry_delay_sec: float = 1.0,
        request_timeout_sec: float = 120.0,
        logger=None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = self.MODEL_ALIAS.get(model_name, model_name)
        self.max_workers = max(1, int(max_workers))
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.request_timeout_sec = max(1.0, float(request_timeout_sec))
        self.logger = logger
        self._stats_lock = Lock()
        self._stats = {
            "requests": 0,
            "retries": 0,
            "failures": 0,
        }

    def _update_stats(self, retries_used: int, failed: bool) -> None:
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["retries"] += max(0, retries_used)
            if failed:
                self._stats["failures"] += 1

    def get_stats_snapshot(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def _generate_one(
        self,
        prompt: str,
        idx: int,
        total: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.request_timeout_sec,
                )
                text = (response.choices[0].message.content or "").strip()
                self._update_stats(retries_used=attempt - 1, failed=False)
                if self.logger:
                    self.logger.info(
                        "[VLLMGenerator] generated %s/%s\n[INPUT PROMPT]\n%s\n[OUTPUT]\n%s",
                        idx + 1,
                        total,
                        prompt,
                        text,
                    )
                return text
            except Exception as exc:  # pragma: no cover - network path
                if self.logger:
                    self.logger.warning(
                        "[VLLMGenerator] failed (%s/%s) for prompt %s: %s",
                        attempt,
                        self.max_retries,
                        idx + 1,
                        exc,
                    )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)

        self._update_stats(retries_used=self.max_retries - 1, failed=True)
        return "__VLLM_REQUEST_FAILED__"

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            return []

        total = len(prompts)
        if total == 1 or self.max_workers == 1:
            return [
                self._generate_one(
                    prompt=prompts[0],
                    idx=0,
                    total=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ] if total == 1 else [
                self._generate_one(
                    prompt=prompt,
                    idx=idx,
                    total=total,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                for idx, prompt in enumerate(prompts)
            ]

        responses: List[str] = [""] * total
        worker_count = min(self.max_workers, total)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    prompt,
                    idx,
                    total,
                    max_tokens,
                    temperature,
                    top_p,
                ): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    responses[idx] = future.result()
                except Exception:  # pragma: no cover - safety path
                    self._update_stats(retries_used=self.max_retries - 1, failed=True)
                    responses[idx] = "__VLLM_REQUEST_FAILED__"

        return responses
