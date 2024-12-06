import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

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
    'DL'  # Target column
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

# Define models
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
knn_clf = KNeighborsClassifier()

# Hyperparameter tuning using GridSearchCV
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

param_grid_knn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(knn_clf, param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_

# Train the Logistic Regression model
log_reg.fit(X_train_scaled, y_train)

# Evaluate models using cross-validation for robustness
models = {
    'Logistic Regression': log_reg,
    'Random Forest': best_rf,
    'K-Nearest Neighbors': best_knn
}

results = {}
for name, model in models.items():
    if name == 'Random Forest':  # RF doesn't need scaling
        X_train_used = X_train
    else:
        X_train_used = X_train_scaled

    cv_scores = cross_val_score(model, X_train_used, y_train, cv=5, scoring='f1')
    model.fit(X_train_used, y_train)
    preds = model.predict(X_test_scaled if name != 'Random Forest' else X_test)
    results[name] = {
        'CV F1-Score': cv_scores.mean(),
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds)
    }

# Print results in a table format
results_df = pd.DataFrame(results).T
print("Model Performance Metrics:")
print(results_df)

# Save the best model based on F1-Score
best_model_name = results_df['F1-Score'].idxmax()
best_model = models[best_model_name]

# Save the best model and the scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"Best model saved: {best_model_name}")
