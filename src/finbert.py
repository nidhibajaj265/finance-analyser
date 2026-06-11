from transformers import pipeline
from src.config import HF_TOKEN

#Need to add HF_TOKEN here
finbert = pipeline("sentiment-analysis",model="ProsusAI/finbert",token=HF_TOKEN)

def call_finbert(text: str) -> dict:
    # Using [0] here to refer the first element in the list 
    # as finbert is returning a list of dict intead of a dict
    return finbert(text)[0] 