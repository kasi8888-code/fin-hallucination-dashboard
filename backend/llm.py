import os
import time

from groq import Groq


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def call_llm(prompt: str, max_retries: int = 3) -> str:

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
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
