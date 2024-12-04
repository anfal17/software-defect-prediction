import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # To save the trained model

# Create a dummy dataset
data = {
    'LOC': [20, 50, 10, 30, 40, 15],
    'Cyclomatic Complexity': [5, 10, 3, 6, 7, 4],
    'Methods': [2, 3, 1, 2, 4, 2],
    'Buggy': [1, 0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Split into features and target
X = df[['LOC', 'Cyclomatic Complexity', 'Methods']]  # Features
y = df['Buggy']  # Target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=min(len(X_train), 3))
}

# Train models and calculate metrics
scores = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    scores[model_name] = {
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }
    # print(scores)

# Save the best model (e.g., Random Forest)
best_model = models['Random Forest']
joblib.dump(best_model, 'bug_prediction_model.pkl')  # Save the model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
