
import numpy as np
from textblob import TextBlob
import re

def extract_features(text):
    text = str(text)
    features = []

    # Character count
    features.append(len(text))

    # Word count
    words = text.split()
    features.append(len(words))

    # Exclamation marks
    features.append(text.count('!'))

    # Capital words
    features.append(sum(1 for word in words if word.isupper()))

    # Sentiment polarity
    blob = TextBlob(text)
    features.append(blob.sentiment.polarity)

    # Lexical diversity
    features.append(len(set(words)) / (len(words) + 1))

    # NEW: Adjective ratio (basic heuristic using POS from TextBlob)
    try:
        tags = blob.tags
        adjectives = [word for word, tag in tags if tag.startswith('JJ')]
        features.append(len(adjectives) / (len(words) + 1))
    except:
        features.append(0)

    # NEW: Sentence length variance
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    if len(sentence_lengths) > 1:
        features.append(np.var(sentence_lengths))
    else:
        features.append(0)

    return features
