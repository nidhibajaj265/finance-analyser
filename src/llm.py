from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, LLM_MODEL

client = InferenceClient(
    api_key=HF_TOKEN
)

def call_llm(user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=250
    )

    return completion.choices[0].message.content