import numpy as np
from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import joblib
import os

os.makedirs("Model", exist_ok=True)

# LOAD PREPROCESSED DATA
X = load_npz("../Task2/Output/X_train.npz")
y = np.load("../Task2/Output/y_train.npy")

print("\n=====================================================")
print("TASK 3 - MODEL SELECTION")
print("=====================================================")

print("Training feature shape:", X.shape)
print("Training label shape:", y.shape)

# DEFINE MODELS
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=150, # Number of decision trees
        max_depth=20, # Maximum tree depth
        random_state=42 # Reproducible results
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=3000, # Maximum training iterations
        solver="saga", # Optimization algorithm
        class_weight="balanced", # Handles class imbalance
        random_state=42 # Reproducible results
    )
}

# MODEL EVALUATION
best_model = None
best_score = 0

print("\n=====================================================")
print("MODEL PERFORMANCE")
print("=====================================================")

for name, model in models.items():

    score = cross_val_score(

        model,
        X,
        y,

        cv=3,
        # 3-fold cross validation

        scoring="f1_macro"
        # Evaluation metric

    ).mean()

    print(f"{name}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model

# BEST MODEL
print("\n=====================================================")
print("BEST MODEL SELECTION")
print("=====================================================")

print("Best Score:", round(best_score, 4))
print("Selected Model:", best_model)

# TRAIN BEST MODEL
best_model.fit(X, y)

print("\nBest model trained successfully.")

# SAVE MODEL
joblib.dump(best_model, "Model/best_model.pkl")

print("\n=====================================================")
print("TASK 3 COMPLETED")
print("BEST MODEL SAVED")
print("=====================================================")