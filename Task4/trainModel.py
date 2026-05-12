import numpy as np
from scipy.sparse import load_npz

import joblib
import os

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

# SETUP
os.makedirs("Model", exist_ok=True)

# LOAD DATA
X_train = load_npz("../Task2/Output/X_train.npz")
X_test = load_npz("../Task2/Output/X_test.npz")

y_train = np.load("../Task2/Output/y_train.npy")
y_test = np.load("../Task2/Output/y_test.npy")

print("\n=====================================================")
print("TASK 4 - MODEL TRAINING AND EVALUATION")
print("=====================================================")

print("Training feature shape:", X_train.shape)
print("Testing feature shape :", X_test.shape)

# LOAD BEST MODEL
model = joblib.load("../Task3/Model/best_model.pkl")

print("\n=====================================================")
print("BEST MODEL LOADED")
print("=====================================================")

print(model)

# TRAIN MODEL
model.fit(X_train, y_train)

print("\nModel training completed successfully.")

# PREDICTIONS
pred = model.predict(X_test)

print("\n=====================================================")
print("MODEL PREDICTIONS COMPLETED")
print("=====================================================")

# ACCURACY SCORE
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy Score:")
print(round(accuracy, 4))

# CLASSIFICATION REPORT
print("\n=====================================================")
print("CLASSIFICATION REPORT")
print("=====================================================")

report = classification_report(y_test, pred)

print(report)

# SAVE TRAINED MODEL
joblib.dump(model, "Model/trained_model.pkl")
