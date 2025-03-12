import pandas as pd
import numpy as np

import pickle
import os

from sklearn.preprocessing import MinMaxScaler

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

scaler = MinMaxScaler().fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

scaler_path = "artifacts/scalers"
os.makedirs(scaler_path, exist_ok=True) # Create processed folder if doesn't exists
scaler_path = os.path.join(scaler_path, "minmaxscaler.pkl")

with open(scaler_path, mode="wb") as f:
    pickle.dump(scaler, f)

print(f"MinMaxScaler saved to {scaler_path}")

processed_path = "data/processed_data"
os.makedirs(processed_path, exist_ok=True) # Create processed folder if doesn't exists

X_train_scaled.to_csv(os.path.join(processed_path, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(processed_path, "X_test_scaled.csv"), index=False)

print(f"Scaled X_train and X_test saved to {processed_path}")