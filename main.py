# File: fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\user\Downloads\CLOUDCREDITS\Fraud_detection\data\creditcard.csv')

# Step 2: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Separate features and target
X = data.drop('Class', axis=1)  # Features (transactions data)
y = data['Class']  # Target (1 = Fraudulent, 0 = Normal)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Model Training and Evaluation
# Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Model Prediction
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Step 7: Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate Logistic Regression
evaluate_model(y_test, y_pred_lr, 'Logistic Regression')

# Evaluate Random Forest
evaluate_model(y_test, y_pred_rf, 'Random Forest')

# Step 8: ROC Curve and AUC
def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC Curve
plot_roc_curve(lr_model, X_test, y_test, 'Logistic Regression')
plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')

# Step 9: Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_test)

# Convert Isolation Forest predictions to 0 (Normal) and 1 (Fraud)
y_pred_iso = [1 if p == -1 else 0 for p in y_pred_iso]

# Evaluate Isolation Forest
evaluate_model(y_test, y_pred_iso, 'Isolation Forest')

# Step 10: Visualize the distribution of normal vs. fraudulent transactions
plt.figure(figsize=(6,4))
sns.countplot(data['Class'])
plt.title('Distribution of Normal vs Fraudulent Transactions')
plt.xlabel('Class (0 = Normal, 1 = Fraud)')
plt.ylabel('Count')
plt.show()
