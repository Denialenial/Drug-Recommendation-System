import numpy as np
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import joblib
import os

os.makedirs("Model", exist_ok=True)

# LOAD DATA
X = load_npz("../Task2/Output/X_train.npz")
y = np.load("../Task2/Output/y_train.npy")

print("\n=====================================================")
print("TASK 5 - HYPERPARAMETER TUNING")
print("=====================================================")

print("Feature matrix shape:", X.shape)
print("Label shape:", y.shape)

# PARAMETER GRID
params = {

    "C": [0.5, 1.0, 2.0],
    # Regularization strength

    "max_iter": [5000]
    # Maximum training iterations
}

print("\n=====================================================")
print("PARAMETER GRID")
print("=====================================================")

print(params)

# GRID SEARCH SETUP
model = LogisticRegression(

    solver="saga",
    class_weight="balanced"
)

grid = GridSearchCV(

    estimator=model,

    param_grid=params,

    cv=3,
    # 3-fold cross validation

    n_jobs=-1,
    # Uses all CPU cores

    scoring="f1_macro"
)

# TRAIN GRID SEARCH
print("\n=====================================================")
print("GRID SEARCH TRAINING STARTED")
print("=====================================================")

grid.fit(X, y)

print("\nGrid search completed successfully.")

# BEST RESULTS
print("\n=====================================================")
print("BEST PARAMETERS")
print("=====================================================")

print(grid.best_params_)

print("\n=====================================================")
print("BEST SCORE")
print("=====================================================")

print(round(grid.best_score_, 4))

# SAVE BEST MODEL
joblib.dump(

    grid.best_estimator_,
    "Model/tuned_model.pkl"
)
