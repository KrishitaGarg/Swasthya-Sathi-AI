import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ===============================
# Training on Data (Training.csv)
# ===============================

# Load Training Dataset
df = pd.read_csv("data/Training.csv")

# Check if 'prognosis' column exists
if "prognosis" not in df.columns:
    raise ValueError("Error: 'prognosis' column not found in dataset.")

# Handle Missing Values (Replace NaNs with 0)
df.fillna(0, inplace=True)

# Encode target variable (disease)
label_encoder = LabelEncoder()
df["prognosis"] = label_encoder.fit_transform(df["prognosis"])

# Feature Selection
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Ensure there are enough samples
if len(X) == 0 or len(y) == 0:
    raise ValueError("Error: No data available for training.")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")
print("Label encoder saved as label_encoder.pkl")

# Save symptom list
joblib.dump(X.columns.tolist(), "symptom_list.pkl")
print("Symptom list saved as symptom_list.pkl")

# ================================
# Testing on New Data (Testing.csv)
# ================================

print("\nEvaluating models on test data...")

# Load Testing Dataset
test_df = pd.read_csv("data/Testing.csv")

# Drop unnamed extra columns if present
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Handle Missing Values in Testing Data
test_df.fillna(0, inplace=True)

# Ensure 'prognosis' column exists
if "prognosis" not in test_df.columns:
    raise ValueError("Error: 'prognosis' column not found in Testing.csv.")

# Encode target variable using the same LabelEncoder
test_df["prognosis"] = label_encoder.transform(test_df["prognosis"])

# Ensure test data has the same features as training data
common_columns = X.columns.intersection(test_df.columns)  # Only select matching columns
X_test_real = test_df[common_columns]

# If test data is missing any training features, add them as zeros
for col in X.columns:
    if col not in X_test_real.columns:
        X_test_real[col] = 0  # Add missing columns with default value

y_test_real = test_df["prognosis"]

# Evaluate each model
for name, model in models.items():
    y_pred = model.predict(X_test_real)
    accuracy = accuracy_score(y_test_real, y_pred) * 100
    print(f"📈 {name} Accuracy: {accuracy:.2f}%")

print("\n Model training and evaluation complete!")
