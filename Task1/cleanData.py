import pandas as pd
import html
import os

os.makedirs('Clean Data', exist_ok=True)

# LOAD DATA
train_raw = pd.read_excel('Raw Data/drugsComTrain_raw.xlsx')
test_raw = pd.read_csv('Raw Data/drugsComTest_raw.csv')

print("\n=====================================================")
print("DATA LOADED")
print("=====================================================")

print(f"Train shape (raw): {train_raw.shape}")
print(f"Test shape (raw): {test_raw.shape}")

# MISSING VALUES (BEFORE CLEANING)
print("\n=====================================================")
print("MISSING VALUES (BEFORE CLEANING)")
print("=====================================================")

print("\nTrain missing values per column:")
print(train_raw.isnull().sum())

print("\nTotal rows with missing values (Train):",
      train_raw.isnull().any(axis=1).sum())

print("\nTest missing values per column:")
print(test_raw.isnull().sum())

print("\nTotal rows with missing values (Test):",
      test_raw.isnull().any(axis=1).sum())

# DROP CRITICAL MISSING VALUES
train = train_raw.dropna(subset=['drugName', 'condition', 'review'])
test = test_raw.dropna(subset=['drugName', 'condition', 'review'])

print("\n=====================================================")
print("AFTER DROPPING CRITICAL MISSING VALUES")
print("=====================================================")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# FILL NUMERIC MISSING VALUES
train['rating'] = train['rating'].fillna(train['rating'].median())
test['rating'] = test['rating'].fillna(train['rating'].median())

train['usefulCount'] = train['usefulCount'].fillna(0)
test['usefulCount'] = test['usefulCount'].fillna(0)

# TEXT CLEANING
train['review'] = train['review'].apply(html.unescape)
test['review'] = test['review'].apply(html.unescape)

# FILTER CONDITIONS
conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']

train = train[train['condition'].isin(conditions)]
test = test[test['condition'].isin(conditions)]

print("\n=====================================================")
print("AFTER CONDITION FILTERING")
print("=====================================================")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# OUTLIER REMOVAL FUNCTION
def remove_outliers(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    before = df.shape[0]

    df_cleaned = df[(df[column] >= lower) & (df[column] <= upper)]

    after = df_cleaned.shape[0]

    print("\n-----------------------------------------------------")
    print(f"OUTLIERS REMOVAL REPORT: {column}")
    print("-----------------------------------------------------")
    print(f"Before: {before}")
    print(f"After : {after}")
    print(f"Removed: {before - after}")

    return df_cleaned

# OUTLIERS (RATING)
train = remove_outliers(train, 'rating')
test = remove_outliers(test, 'rating')

# OUTLIERS (USEFUL COUNT)
train = remove_outliers(train, 'usefulCount')
test = remove_outliers(test, 'usefulCount')

# FINAL SHAPE
print("\n=====================================================")
print("FINAL CLEANED DATASET SHAPE")
print("=====================================================")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

print("\nMissing values after cleaning:")
print(train.isnull().sum().sum())

# SAVE DATA
train.to_csv('Clean Data/train_cleaned.csv', index=False)
test.to_csv('Clean Data/test_cleaned.csv', index=False)

print("\n=====================================================")
print("TASK 1 COMPLETED - CLEAN DATA SAVED")
print("=====================================================")