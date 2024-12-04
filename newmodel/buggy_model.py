import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Import joblib for saving models and scalers

# Load the dataset
data = pd.read_csv('extracted_features.csv')

# List of relevant columns (input features and target)
columns_of_interest = [
    'avgCYCLOMATIC_COMPLEXITY',
    'NUM_OF_CHILDREN',
    'DEP_ON_CHILD',
    'PERCENT_PUB_DATA',
    'avgLOC_TOTAL',
    'avgLOC_EXECUTABLE',
    'COUPLING_BETWEEN_OBJECTS',
    'avgHALSTEAD_EFFORT',
    'DL'  # This seems to be the target column
]

# Extract only the relevant columns
extracted_data = data[columns_of_interest]

# Convert boolean-like string values ('_TRUE', 'TRUE', 'FALSE') to numeric (1, 0) if present
extracted_data['DL'] = extracted_data['DL'].replace({'_TRUE': 1, 'TRUE': 1, 'FALSE': 0})

# Handle missing values by replacing them with column means
extracted_data = extracted_data.fillna(extracted_data.mean(numeric_only=True))

# Split the dataset into features (X) and target (y)
X = extracted_data.drop(columns=['DL'])  # Features
y = extracted_data['DL']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for algorithms like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
knn_clf = KNeighborsClassifier()

# Train the models
log_reg.fit(X_train_scaled, y_train)
rf_clf.fit(X_train, y_train)  # Random Forest doesn't require feature scaling
knn_clf.fit(X_train_scaled, y_train)

# Predict on the test data
log_reg_preds = log_reg.predict(X_test_scaled)
rf_clf_preds = rf_clf.predict(X_test)
knn_clf_preds = knn_clf.predict(X_test_scaled)

# Evaluate the models
def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

log_reg_metrics = evaluate_model(y_test, log_reg_preds)
rf_clf_metrics = evaluate_model(y_test, rf_clf_preds)
knn_clf_metrics = evaluate_model(y_test, knn_clf_preds)

# Print the metrics for each model
print("Logistic Regression Metrics:", log_reg_metrics)
print("Random Forest Metrics:", rf_clf_metrics)
print("K-Nearest Neighbors Metrics:", knn_clf_metrics)

# Save the best model based on F1-Score (you can also select by other metrics)
metrics = {
    'Logistic Regression': log_reg_metrics,
    'Random Forest': rf_clf_metrics,
    'K-Nearest Neighbors': knn_clf_metrics
}

# Select the best model based on F1-Score
best_model_name = max(metrics, key=lambda model: metrics[model]['F1-Score'])
best_model = None

if best_model_name == 'Logistic Regression':
    best_model = log_reg
elif best_model_name == 'Random Forest':
    best_model = rf_clf
elif best_model_name == 'K-Nearest Neighbors':
    best_model = knn_clf

# Save the best model and the scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

print(f"Best model and scaler saved.")
