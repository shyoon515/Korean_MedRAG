import os
import time
import importlib.util
from pathlib import Path
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
        logger=None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = self.MODEL_ALIAS.get(model_name, model_name)
        self.logger = logger

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        responses: List[str] = []
        for idx, prompt in enumerate(prompts):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = (response.choices[0].message.content or "").strip()
            responses.append(text)
            if self.logger:
                self.logger.info(
                    "[VLLMGenerator] generated %s/%s\n[INPUT PROMPT]\n%s\n[OUTPUT]\n%s",
                    idx + 1,
                    len(prompts),
                    prompt,
                    text,
                )

        return responses
