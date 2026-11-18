
import os
from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def test_inference(token):
    client = InferenceClient(
        token=token
    )

    response = client.chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. "
                    "Answer briefly."
                )
            },
            {
                "role": "user",
                "content": (
                    "Which city is the Eiffel Tower in?"
                )
            }
        ],
        max_tokens=100,
        temperature=0
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "Set HF_TOKEN as an environment variable."
        )

    print(test_inference(token))
