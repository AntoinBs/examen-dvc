import pandas as pd
import numpy as np

import os

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw_data/raw.csv')

df = df.drop('date', axis=1)

X = df.drop("silica_concentrate", axis=1)
y = df['silica_concentrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

processed_path = "data/processed_data"
os.makedirs(processed_path, exist_ok=True) # Create processed folder if doesn't exists

X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

print("Train/test datasets have been created from data/raw_data/raw.csv and saved into data/processed_data folder")