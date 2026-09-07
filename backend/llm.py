import os
import time

from groq import Groq


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
                temperature=temperature,
		        max_tokens =300
            )

            return response.choices[0].message.content

        except Exception as e:

            # Do not retry permanent request errors
            if "413" in str(e):
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 * (2* attempt + 1)

            print(
                f"LLM temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)
