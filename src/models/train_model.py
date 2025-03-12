import pandas as pd
import numpy as np

import pickle

from sklearn.svm import SVR

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

best_params_path = "models/best_params.pkl"
with open(best_params_path, "rb") as f:
    params = pickle.load(f)

clf = SVR(**params)
clf.fit(X_train, y_train.values.ravel())

model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print('Model have been trained with params:', params)
print('Model have been saved to:', model_path)