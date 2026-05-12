import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

train = pd.read_csv('Clean Data/train_cleaned.csv')

os.makedirs("Graphs", exist_ok=True)

# RATING STATISTICS
print("\n================ RATING STATISTICS ================")
print(train['rating'].describe())

# CONDITION DISTRIBUTION
print("\n================ CONDITION DISTRIBUTION ================")

condition_counts = train['condition'].value_counts()
print(condition_counts)

print("\nCounts per condition:")
for condition, count in condition_counts.items():
    print(condition, ":", count)

# CORRELATION
print("\n================ CORRELATION =================")

print(train[['rating', 'usefulCount']].corr())

# DATASET SUMMARY
print("\n================ DATASET SUMMARY ================")

print("Total records:", len(train))
print("Total columns:", train.shape[1])
print("Missing values:", train.isnull().sum().sum())

# GRAPH 1 - HISTOGRAM
plt.figure(figsize=(8,5))
plt.hist(train['rating'], bins=10, edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")

plt.savefig("Graphs/rating_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 2 - BAR CHART
plt.figure(figsize=(8,5))
train['condition'].value_counts().plot(kind='bar')
plt.title("Condition Frequency")
plt.xlabel("Condition")
plt.ylabel("Count")

plt.savefig("Graphs/condition_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 3 - AVERAGE RATING
plt.figure(figsize=(8,5))
train.groupby('condition')['rating'].mean().plot(kind='bar')
plt.title("Average Rating per Condition")
plt.xlabel("Condition")
plt.ylabel("Average Rating")

plt.savefig("Graphs/avg_rating_condition.png", dpi=300, bbox_inches='tight')
plt.close()

# GRAPH 4 - PIE CHART
plt.figure(figsize=(7,7))
plt.pie(
    condition_counts,
    labels=condition_counts.index,
    autopct='%1.1f%%',
    startangle=140
)

plt.title("Condition Distribution")

plt.savefig("Graphs/condition_pie_chart.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll graphs saved in Task1/Graphs/")