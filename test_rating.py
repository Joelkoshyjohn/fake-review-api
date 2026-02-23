from rating_anomaly import rating_anomaly_score

ratings = [
    3, 4, 3, 3, 4, 3, 2, 4, 3, 3,
    3, 4, 3, 3, 4, 3, 3, 4, 3, 3,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5
]

score, flag = rating_anomaly_score(ratings)

print("Anomaly Score:", round(score, 3))

if flag:
    print("⚠ Rating Spike Detected")
else:
    print("Rating Pattern Normal")