from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

#Used [] for HF_TOKEN here because we want it to throw an error 
# if the environment variable is not set, 
# rather than defaulting to None or an empty string. 
# This way, we can catch the error early
# and ensure that the necessary configuration is provided before 
# running the application.
HF_TOKEN = os.environ.get("HF_TOKEN")

#Used () here for OPENAI_API_KEY because we dont want it to throw an error 
# if the environment variable is not set and default to None or an empty string. 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SEC_EDGAR_USER_AGENT = "finance-analyser nidi.bajaj@gmail.com"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = str(PROJECT_ROOT/"chroma_db")
ARTICLES_CACHE_PATH = str(PROJECT_ROOT/"data")
PROCESSED_ARTICLES_PATH = str(PROJECT_ROOT/"data"/"processed")
SIGNALS_PATH = str(PROJECT_ROOT/"data"/"signals")
os.makedirs(PROCESSED_ARTICLES_PATH, exist_ok=True)
os.makedirs(SIGNALS_PATH, exist_ok=True)
DEFAULT_SYSTEM_PROMPT_LLM = """You are an expert financial alaysist, 
                        that reads the finance related articles and give back 
                        response based on user prompts"""