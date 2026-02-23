from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def check_similarity(new_vector, existing_vectors, threshold=0.85):
    similarities = cosine_similarity(new_vector, existing_vectors)
    max_similarity = np.max(similarities)

    if max_similarity > threshold:
        return max_similarity, True
    else:
        return max_similarity, False