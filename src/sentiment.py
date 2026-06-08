from nltk import sent_tokenize
from huggingface_hub import InferenceClient
from src.llm import call_llm

def generate_sentiment_report(articles: list[dict]) -> dict:

    for article in articles:
        if article.get('text','') != '':
            text_to_analyse = article.get('text','')
        else:
            text_to_analyse = article.get('summary','') + " " + article.get('title','');
            
        if text_to_analyse.strip() == '':
            article['sentiment'] = 'neutral'

        sents_in_article = sent_tokenize(text_to_analyse)

        for sentence in sents_in_article:
            sentiment = call_llm(sentence)
            print(f"Sentence: {sentence}\nSentiment: {sentiment}\n")
