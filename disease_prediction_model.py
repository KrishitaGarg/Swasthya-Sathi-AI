import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import joblib
import os

# Load training dataset
dataset_file = "data/Training.csv"
dataframe = pd.read_csv(dataset_file)
dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
dataframe.fillna(0, inplace=True)

# Features and target selection
X = dataframe.drop(columns=['prognosis'])  # Symptom columns
y = dataframe['prognosis']  # Disease labels

# Encode disease labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Hyperparameter tuning for Decision Tree
param_dist = {
    'max_depth': randint(10, 80),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']
}
clf = DecisionTreeClassifier()
random_search = RandomizedSearchCV(clf, param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)
bestModelDT = random_search.best_estimator_

# Decision Tree evaluation
predictions = bestModelDT.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, predictions))

# Hyperparameter tuning for Random Forest
param_dist_rf = {
    'n_estimators': randint(50, 200),
    'max_features': ['auto', 'sqrt'],
    'max_depth': randint(10, 80),
    'min_samples_split': randint(2, 50),
    'min_samples_leaf': randint(1, 50),
    'bootstrap': [True, False]
}
rf_clf = RandomForestClassifier()
rf_random_search = RandomizedSearchCV(rf_clf, param_dist_rf, n_iter=50, cv=5, random_state=42)
rf_random_search.fit(X_train, y_train)
bestModelRF = rf_random_search.best_estimator_

# Random Forest evaluation
predictionRF = bestModelRF.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, predictionRF))

# Load testing dataset
test_dataset_file = "data/Testing.csv"
test_dataframe = pd.read_csv(test_dataset_file)
test_dataframe = test_dataframe.loc[:, ~test_dataframe.columns.str.contains('^Unnamed')]
test_dataframe.fillna(0, inplace=True)

# Transform test labels using saved LabelEncoder
test_dataframe['prognosis'] = label_encoder.transform(test_dataframe['prognosis'])
X_test_real = test_dataframe.drop(columns=['prognosis'])
y_test_real = test_dataframe['prognosis']

# Evaluate models on test data
print("\nEvaluating models on real test data...")
final_predictions_DT = bestModelDT.predict(X_test_real)
final_predictions_RF = bestModelRF.predict(X_test_real)
print("Decision Tree Model on Test Data:")
print(classification_report(y_test_real, final_predictions_DT))
print("Random Forest Model on Test Data:")
print(classification_report(y_test_real, final_predictions_RF))

# Visualization of disease distribution
plt.figure(figsize=(12, 5))
plt.xticks(rotation=90)
sns.countplot(y=dataframe['prognosis'])
plt.xlabel('Count')
plt.ylabel('Disease')
plt.title('Number of Cases per Disease')
plt.savefig("disease_distribution.png")
plt.close()

# Function to plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(model_name + "_confusion_matrix.png")
    plt.close()

# Plot confusion matrices for both models
plot_confusion_matrix(y_test_real, final_predictions_DT, "Decision Tree")
plot_confusion_matrix(y_test_real, final_predictions_RF, "Random Forest")

# Save trained models and metadata
joblib.dump(bestModelDT, os.path.join("models/", "DecisionTreeModel.pkl"))
joblib.dump(bestModelRF, os.path.join("models/", "RandomForestModel.pkl"))
joblib.dump(label_encoder, os.path.join("models/", "LabelEncoder.pkl"))
joblib.dump(X.columns.tolist(), os.path.join("models/", "SymptomList.pkl"))

print("Models and metadata saved successfully!")
