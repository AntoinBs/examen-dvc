import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

params_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'degree' : [2, 3, 4],
    'gamma' : ['scale', 'auto']
}

clf = GridSearchCV(estimator=SVR(), param_grid=params_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
clf.fit(X_train, y_train.values.ravel())

best_params = clf.best_params_

best_params_path = "models/best_params.pkl"
with open(best_params_path, "wb") as f:
    pickle.dump(best_params, f)

print("Best parameters for SVR model are :", best_params)
print("It's been saved to:", best_params_path)
print("The score is : RMSE =", -clf.best_score_)
