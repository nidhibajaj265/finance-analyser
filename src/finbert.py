from src.config import HF_TOKEN
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import ConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

client = InferenceClient(api_key=HF_TOKEN)

def _log_retry(retry_state):
    logger.warning(
        f"FinBERT call failed ({retry_state.outcome.exception()}), "
        f"retrying attempt {retry_state.attempt_number}..."
    )

@retry(
    retry=retry_if_exception_type((HfHubHTTPError, ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=_log_retry,
)
def call_finbert(text: str) -> dict:
    result = client.text_classification(text, model="ProsusAI/finbert")
    logger.debug(f"FinBERT result: {result[0]}")
    return result[0]