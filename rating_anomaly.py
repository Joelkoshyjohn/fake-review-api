import numpy as np

def rating_anomaly_score(ratings, window_size=10, threshold=2.5):
    """
    ratings: list of historical ratings (chronological order)
    window_size: number of recent ratings to analyze
    threshold: z-score threshold for anomaly detection
    """

    if len(ratings) < window_size + 5:
        return 0, False  # not enough history

    ratings = np.array(ratings)

    historical_mean = np.mean(ratings)
    historical_std = np.std(ratings)

    if historical_std == 0:
        return 0, False

    recent_avg = np.mean(ratings[-window_size:])

    z_score = (recent_avg - historical_mean) / historical_std

    if abs(z_score) > threshold:
        return abs(z_score), True
    else:
        return abs(z_score), False