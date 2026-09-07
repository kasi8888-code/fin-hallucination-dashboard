"""
LLMSampler: SelfCheckGPT-style multi-sample response generator ("The Brain"
role owns the sampling protocol that feeds the variance/entropy statistics).

The central idea of SelfCheckGPT (Manakul et al. 2023) is that a model that
*knows* the answer should produce consistent responses across repeated
independent samples, whereas a hallucinating model produces high-variance,
contradictory outputs. Sampling must therefore be:

  * independent  - each sample uses a fresh response (no prior samples in context)
  * stochastic   - temperature >= 0.7 so the sampler explores the answer space
  * N > 1        - enough samples to estimate variance (default N = 5)

Backends:
  - "openai":   Chat Completions (gpt-4o-mini default)
  - "groq":     Groq's fast OpenAI-compatible endpoint (qwen/qwen3.8-27b)
  - "ollama":   local OpenAI-compatible endpoint (localhost:11434)

`provider="auto"` resolution order: OPENAI_API_KEY -> GROQ_API_KEY -> local
Ollama, so a GROQ_API_KEY set in `.env` is picked up automatically.

An optional single "deterministic" pass (temperature ~ 0) can be requested and
is used as the primary response that gets compared against a trusted document.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Availability flags (dependencies are optional by design)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on local env
    import openai  # noqa: F401
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:  # pragma: no cover
    import httpx
    HAS_HTTPX = True
except Exception:
    HAS_HTTPX = False

_FINANCIAL_SYSTEM_PROMPT = (
    "You are a precise financial analyst. Answer the user's question using only "
    "verifiable facts. State figures with their units and time period. If you do "
    "not know the exact figure, say so. Keep the answer to 2-4 sentences."
)


def _load_dotenv(path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (only sets keys
    that are not already present). Zero-dependency substitute for python-dotenv,
    sufficient for GROQ_API_KEY / OPENAI_API_KEY / OLLAMA_HOST."""
    if path is None:
        candidates = [".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]
        for candidate in candidates:
            if os.path.isfile(candidate):
                path = candidate
                break
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:  # pragma: no cover - best-effort loader
        return


# Load .env once at import time so provider="auto" sees GROQ_API_KEY.
_load_dotenv()


@dataclass
class SamplingResult:
    """Container for one full SelfCheckGPT sampling episode."""

    question: str
    sampled_responses: List[str]
    deterministic_response: Optional[str] = None
    provider: str = ""
    model: str = ""
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "provider": self.provider,
            "model": self.model,
            "config": self.config,
            "deterministic_response": self.deterministic_response,
            "sampled_responses": self.sampled_responses,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)


def _is_retryable(exc: Exception) -> bool:
    if HAS_OPENAI and isinstance(exc, (openai.RateLimitError, openai.APITimeoutError)):  # noqa: F821
        return True
    if hasattr(exc, "status_code") and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


def _chat_with_openai_compatible(client, *, model, question, system_prompt, temperature, max_tokens):
    """Single chat completion via any OpenAI-compatible client (OpenAI or Groq)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


class LLMSampler:
    """Generates independent stochastic samples from an LLM for self-consistency scoring."""

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        system_prompt: str = _FINANCIAL_SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries

        if provider == "auto":
            # Provider priority: OPENAI_API_KEY, then GROQ_API_KEY, then local ollama.
            if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            else:
                self.provider = "ollama"

        if self.provider == "groq":
            # Groq is OpenAI-compatible but with its own key + base URL. We reuse
            # the openai client pointed at Groq. Note: Groq clamps temperature to
            # a minimum of 1e-8, so "temperature=0" is not literally 0 - good
            # enough for the deterministic pass.
            if not HAS_OPENAI:
                raise RuntimeError("OpenAI package not installed (needed for the Groq provider). pip install openai")
            from openai import OpenAI

            self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
            self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
            self._client = OpenAI(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                base_url=self.base_url,
            )

        if self.provider not in ("openai", "groq", "ollama"):
            raise ValueError(f"provider must be 'openai', 'groq', 'ollama' or 'auto', got {self.provider!r}")

        if self.provider == "openai":
            if not HAS_OPENAI:
                raise RuntimeError("OpenAI package not installed. pip install openai")
            from openai import OpenAI

            self.model = model or "gpt-4o-mini"
            self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.base_url = None
        elif self.provider == "ollama":
            if not HAS_HTTPX:
                raise RuntimeError("httpx is required for the ollama backend")
            self.model = model or "llama3.1"
            self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
            self._client = None

    # -- single chat call ----------------------------------------------------
    def _chat(self, question: str, temperature: float, max_tokens: int) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider in ("openai", "groq"):
                    # Groq is a drop-in OpenAI-compatible endpoint, so both share
                    # the openai ChatCompletions client (Groq client is already
                    # configured with base_url=https://api.groq.com/openai/v1).
                    return _chat_with_openai_compatible(
                        self._client,
                        model=self.model,
                        question=question,
                        system_prompt=self.system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    resp = httpx.post(
                        f"{self.base_url}/v1/chat/completions",
                        json={
                            "model": self.model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "messages": [
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": question},
                            ],
                        },
                        timeout=180.0,
                    )
                    resp.raise_for_status()
                    return (resp.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:  # noqa: BLE001 - retried with backoff below
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise
                time.sleep(1.0 * (2 ** (attempt - 1)))

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    # -- public sampling API -------------------------------------------------
    def sample_many(
        self,
        question: str,
        n: int = 5,
        temperature: float = 0.8,
        max_tokens: int = 200,
        include_deterministic: bool = True,
    ) -> SamplingResult:
        """
        Produce N independent stochastic samples, optionally preceded by one
        deterministic pass (temperature ~ 0). Each sampled response is generated
        from a fresh conversation so samples are conditionally independent —
        this is what makes disagreement statistically meaningful.
        """
        if n < 2:
            raise ValueError("SelfCheckGPT needs at least n=2 stochastic samples for variance estimation")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {temperature}")

        deterministic_response = None
        if include_deterministic:
            deterministic_response = self._chat(question, temperature=0.0, max_tokens=max_tokens)

        sampled_responses = [
            self._chat(question, temperature=temperature, max_tokens=max_tokens)
            for _ in range(n)
        ]
        sampled_responses = [r for r in sampled_responses if r]

        return SamplingResult(
            question=question,
            sampled_responses=sampled_responses,
            deterministic_response=deterministic_response,
            provider=self.provider,
            model=self.model,
            config={"n": n, "temperature": temperature, "max_tokens": max_tokens},
        )
