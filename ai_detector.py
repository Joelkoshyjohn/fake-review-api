import numpy as np
import re

def ai_likeness_score(text):
    text = str(text)
    score = 0

    # Sentence length uniformity
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]

    if len(sentence_lengths) > 1:
        variance = np.var(sentence_lengths)
        if variance < 5:
            score += 0.3

    # Lexical diversity
    words = text.split()
    if len(words) > 0:
        diversity = len(set(words)) / len(words)
        if diversity < 0.4:
            score += 0.2

    # Low punctuation randomness
    if text.count("!") == 0 and text.count("?") == 0:
        score += 0.1

    # Repetition ratio
    if len(words) > 0:
        repetition_ratio = 1 - (len(set(words)) / len(words))
        if repetition_ratio > 0.3:
            score += 0.2

    # Generic opening patterns
    if text.lower().startswith(("this product", "this hotel", "this service")):
        score += 0.2

    return min(score, 1.0)