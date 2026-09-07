"""
Embeddings: semantic-similarity backbone of the NLP "Brain" module.

Implements the cosine-similarity metric used to compare an LLM response against
a trusted source document (factual-consistency scoring):

    similarity = A . B / (||A|| * ||B||)

Backends:
1.  "openai"   - text-embedding-3-small / text-embedding-ada-002 (needs OPENAI_API_KEY)
2.  "groq"     - Groq's OpenAI-compatible embeddings endpoint (needs GROQ_API_KEY;
                 only works if your Groq plan has embedding models enabled)
3.  "ollama"   - local embeddings via Ollama REST API  (e.g. nomic-embed-text)
4.  "offline"  - deterministic word-overlap hashing embedder (fallback; pure numpy,
                 catches exact numeric/entity contradictions without any downloads)

The offline embedder is *not* a semantic-transformer substitute: it is a
trained-lookup-free, TF-style hashed bag-of-words vector used so that the whole
pipeline can run and be tested without network access. For production semantic
scores plug in sentence-transformers (text-embedding) or OpenAI.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backend availability flags (imports are lazy/optional by design)
# ---------------------------------------------------------------------------
HAS_OPENAI = True
try:  # noqa: E402  (openai is optional)
    import openai  # noqa: F401
except Exception:  # pragma: no cover - depends on local env
    HAS_OPENAI = False

HAS_HTTPX = True
try:  # noqa: E402  (httpx used for the ollama REST call; optional)
    import httpx  # noqa: F401
except Exception:  # pragma: no cover
    HAS_HTTPX = False


def _load_dotenv() -> None:
    """Zero-dependency loader for the repo .env (GROQ_API_KEY / OPENAI_API_KEY /
    OLLAMA_HOST). Only sets keys not already present in os.environ."""
    candidates = [".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]
    for path in candidates:
        if not os.path.isfile(path):
            continue
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
        except OSError:  # pragma: no cover - best-effort
            return
        return


_load_dotenv()

_NUM_TOKEN_RE = re.compile(r"[\$]?\b\d+(?:\.\d+)?%?\b")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _numerals(text: str) -> Tuple[frozenset, float]:
    """Return (normalised numeric literals, total). Normalises "81.80" -> "81.8"."""
    nums = set()
    total = 0.0
    for raw in _NUM_TOKEN_RE.findall(text.replace(",", "")):
        try:
            value = float(raw.replace("$", "").replace("%", ""))
        except ValueError:
            continue
        nums.add(f"{value:.6f}")
        total += 1.0
    return frozenset(nums), total


def _hash_token(token: str) -> int:
    """Deterministic 31-bit hash for the offline hashing trick."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _word_tokens(text: str) -> List[str]:
    """Lower-cased alphanumeric word tokens (strips currency/percent markers)."""
    return [t.lower() for t in _WORD_TOKEN_RE.findall(text)]


class OfflineHasherEmbedder:
    """
    Deterministic TF-style hashed bag-of-words embedder with an exact-numeric
    fingerprint channel. No downloads, no randomness, fully reproducible.

    Dims = `dim` word slots + 64 numeral slots. Word weight uses sublinear TF
    (1 + log count) to dampen repeated boilerplate; numerals carry weight 4 so a
    single conflicting figure dominates similarity — mirroring how financial
    hallucinations surface.
    """

    def __init__(self, dim: int = 512, numeric_dim: int = 64):
        self.dim = dim
        self.numeric_dim = numeric_dim
        self.total = dim + numeric_dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.total, dtype=np.float64)
        counts: Dict[str, int] = {}
        for token in _word_tokens(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            slot = _hash_token(token) % self.dim
            vector[slot] += 1.0 + np.log(count)

        numerals, total = _numerals(text)
        for numeral in numerals:
            slot = _hash_token(numeral) % self.numeric_dim
            vector[self.dim + slot] += 4.0 / max(total, 1.0)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.

        similarity = (A . B) / (||A|| * ||B||)

    Returns a value in [-1, 1]; embeddings in this module are non-negative so
    the practical range is [0, 1]. 1.0 = identical direction (semantically
    aligned), 0.0 = orthogonal (no overlap).
    """
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------
class EmbeddingModel:
    """Unified facade over OpenAI / Groq / Ollama / offline hashing embedders."""

    def __init__(
        self,
        backend: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        offline_dim: int = 512,
    ) -> None:
        self.backend = backend
        self.model = model
        self.offline_dim = offline_dim

        if backend == "auto":
            # Provider priority: OPENAI_API_KEY, then GROQ_API_KEY, then offline.
            # NOTE: many Groq keys only enable chat models, not /embeddings, so
            # a GROQ key does NOT auto-select groq here; groq embeddings are an
            # explicit opt-in (`backend="groq"`) that raises a clear error if
            # the key lacks embedding access.
            if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                self.backend = "openai"
            else:
                self.backend = "offline"

        if self.backend == "openai":
            if not HAS_OPENAI:
                raise RuntimeError("OpenAI package not installed. pip install openai")
            from openai import OpenAI  # lazy import keeps startup cheap

            self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model or "text-embedding-3-small"
        elif self.backend == "groq":
            # Groq exposes an OpenAI-compatible /embeddings endpoint on plans
            # where embedding models are enabled. If the key cannot serve them,
            # embed() will raise a clear OpenAI 404 (model_not_found).
            if not HAS_OPENAI:
                raise RuntimeError("OpenAI package not installed (needed for the Groq embedder). pip install openai")
            from openai import OpenAI

            self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
            self._client = OpenAI(
                api_key=api_key or os.getenv("GROQ_API_KEY"),
                base_url=self.base_url,
            )
            self.model = model or os.getenv("GROQ_EMBED_MODEL", "nomic-embed-text-v1.5")
        elif self.backend == "ollama":
            if not HAS_HTTPX:
                raise RuntimeError("httpx is required for the ollama embedder")
            self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
            self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        elif self.backend == "offline":
            self.model = "offline-hashed-tf-v1"
            self._offline = OfflineHasherEmbedder(dim=offline_dim)
        else:
            raise ValueError(f"Unknown embedder backend: {backend!r}")

    @property
    def is_offline(self) -> bool:
        return self.backend == "offline"

    def embed(self, text: str) -> np.ndarray:
        if self.backend in ("openai", "groq"):
            # Groq is a drop-in OpenAI-compatible endpoint, so both share the
            # openai client configured with base_url=https://api.groq.com/openai/v1.
            resp = self._client.embeddings.create(model=self.model, input=text)
            return np.asarray(resp.data[0].embedding, dtype=np.float64)
        if self.backend == "ollama":
            payload = {"model": self.model, "prompt": text}
            resp = httpx.post(f"{self.base_url}/api/embeddings", json=payload, timeout=120.0)
            resp.raise_for_status()
            return np.asarray(resp.json()["embedding"], dtype=np.float64)
        return self._offline.embed(text)

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two raw text strings."""
        return cosine_similarity(self.embed(text_a), self.embed(text_b))
