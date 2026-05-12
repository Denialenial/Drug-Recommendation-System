import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# =====================================================
# SETUP
# =====================================================
os.makedirs("Output", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# =====================================================
# LOAD CLEANED DATA
# =====================================================
train = pd.read_csv("../Task1/Clean Data/train_cleaned.csv")
test = pd.read_csv("../Task1/Clean Data/test_cleaned.csv")

print("\n=====================================================")
print("TASK 2 - DATA LOADED")
print("=====================================================")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# =====================================================
# TEXT CLEANING FUNCTION
# =====================================================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# APPLY CLEANING
# =====================================================
train["clean"] = train["review"].apply(clean)
test["clean"] = test["review"].apply(clean)

print("\n=====================================================")
print("TEXT CLEANING COMPLETED")
print("=====================================================")

print(train[["review", "clean"]].head(3))

# =====================================================
# TF-IDF FEATURE EXTRACTION
# =====================================================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(train["clean"])
X_test = tfidf.transform(test["clean"])

print("\n=====================================================")
print("TF-IDF FEATURE EXTRACTION COMPLETED")
print("=====================================================")

print("Feature matrix shape (Train):", X_train.shape)
print("Feature matrix shape (Test):", X_test.shape)

# =====================================================
# LABEL ENCODING
# =====================================================
le = LabelEncoder()

y_train = le.fit_transform(train["condition"])
y_test = le.transform(test["condition"])

print("\n=====================================================")
print("LABEL ENCODING COMPLETED")
print("=====================================================")

print("Classes:", le.classes_)
print("Encoded labels sample:", np.unique(y_train))

# =====================================================
# SAVE OUTPUTS
# =====================================================
save_npz("Output/X_train.npz", X_train)
save_npz("Output/X_test.npz", X_test)

np.save("Output/y_train.npy", y_train)
np.save("Output/y_test.npy", y_test)

joblib.dump(tfidf, "Model/tfidf.pkl")
joblib.dump(le, "Model/label_encoder.pkl")

print("\n=====================================================")
print("TASK 2 COMPLETED SUCCESSFULLY")
print("OUTPUTS SAVED")
print("=====================================================")