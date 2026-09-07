import os
import time

from groq import Groq


def _load_dotenv() -> None:
    """
    Load KEY=VALUE pairs from the repo-root .env into os.environ.

    The backend is usually started from the ``backend/`` directory (uvicorn
    main:app), while the .env file lives at the repository root, so we search
    upward from this module until a .env is found. Only sets keys that are not
    already present (real environment variables take precedence).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    current = here
    while True:
        candidate = os.path.join(current, ".env")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value.strip().strip('"').strip("'")
            return
        parent = os.path.dirname(current)
        if parent == current:
            return
        current = parent


# Load .env before building the Groq client so GROQ_API_KEY resolves whether
# uvicorn is started from backend/ or from the repository root.
_load_dotenv()


# Single source of truth for the model used by the whole backend.
# NOTE: "groq/compound-mini" currently errors with 413 (Request Entity Too
# Large) on any content prompt, so the working chat model is the default.
DEFAULT_MODEL = "qwen/qwen3.8-27b"


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def call_llm(
    prompt: str,
    temperature: float = 0.8,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> str:

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
            )

            return response.choices[0].message.content

        except Exception as e:

            # Do not retry permanent request errors
            if "413" in str(e):
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 * (attempt + 1)

            print(
                f"LLM temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)
