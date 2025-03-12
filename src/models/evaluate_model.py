import pandas as pd
import numpy as np

import pickle
import json

from sklearn.metrics import root_mean_squared_error, r2_score

X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

model_path = "models/model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE score on test dataset:", rmse)
print("R2 score on test dataset:", r2)

# Predictions saving
y_pred_path = "data/processed_data/y_pred.csv"
y_pred.to_csv(y_pred_path, index=False)
print("Predicted values have been saved to:", y_pred_path)

# Scores saving
score = {
    'rmse': rmse,
    'r2' : r2
}
scores_path = "metrics/scores.json"
with open(scores_path, "w") as f:
    json.dump(score, fp=f)
print("Score of predictions have been saved to:", scores_path)