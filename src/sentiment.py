from nltk import sent_tokenize
from loguru import logger
from src.finbert import call_finbert

def generate_sentiment_report(articles: list[dict]) -> list[dict]:
    logger.info(f"Generating sentiment for {len(articles)} articles...")
    for article in articles:
        article['sentiment'] = 0

        sents_in_article = sent_tokenize(article['text'])
        total_score = 0
        total_sentences = len(sents_in_article)

        for sentence in sents_in_article:
            sentiment = call_finbert(sentence)
            if sentiment.get('label') == 'positive':
                total_score += 1 * sentiment.get('score')

            elif sentiment.get('label') == 'negative':
                total_score += -1 * sentiment.get('score')

        article['sentiment'] = total_score / total_sentences if total_sentences else 0
        logger.debug(f"[{article.get('title', 'untitled')[:50]}] sentiment={article['sentiment']:.3f}")

    logger.info("Sentiment report complete.")
    return articles

if __name__ == '__main__':
    test_articles = [{'text': 'Apple shares surged 5% after record quarterly earnings beat Wall Street expectations.', 'title': '', 'summary': ''}]
    result = generate_sentiment_report(test_articles)
    print(result)
