from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, LLM_MODEL, DEFAULT_SYSTEM_PROMPT_LLM
from huggingface_hub.errors import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

client = InferenceClient(
    api_key=HF_TOKEN
)

def _log_retry(retry_state):
    logger.warning(
        f"LLM call failed ({retry_state.outcome.exception()}), "
        f"retrying attempt {retry_state.attempt_number}..."
    )

@retry(
    retry=retry_if_exception_type((HfHubHTTPError, ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=_log_retry,
)
def call_llm(user_prompt: str, system_prompt:str = DEFAULT_SYSTEM_PROMPT_LLM) -> str:
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=250
    )

    return completion.choices[0].message.content

SUMMARY_SYSTEM_PROMPT = (
    "You are a financial news summariser. Summarise the given article in about 200 words "
    "of clear, plain prose. Focus on the facts and market relevance. Do not add a preamble "
    "or headings — return only the summary text."
)

@retry(
    retry=retry_if_exception_type((HfHubHTTPError, ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=_log_retry,
)
def summarise_article_text(text: str) -> str:
    logger.info(f"Summarising article text ({len(text)} chars)...")
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": text[:6000]},
        ],
        max_tokens=400,
    )
    return completion.choices[0].message.content