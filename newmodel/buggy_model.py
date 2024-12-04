import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Step 1: Load the dataset
df = pd.read_csv('kc1-class-level-defectiveornot.csv')  # Replace with your file path

# Step 2: Extract features and target
features = [
    'PERCENT_PUB_DATA', 'ACCESS_TO_PUB_DATA', 'COUPLING_BETWEEN_OBJECTS', 'DEPTH', 
    'LACK_OF_COHESION_OF_METHODS', 'NUM_OF_CHILDREN', 'DEP_ON_CHILD', 'FAN_IN', 
    'RESPONSE_FOR_CLASS', 'WEIGHTED_METHODS_PER_CLASS', 'minLOC_BLANK', 'minBRANCH_COUNT', 
    'minLOC_CODE_AND_COMMENT', 'minLOC_COMMENTS', 'minCYCLOMATIC_COMPLEXITY', 'minDESIGN_COMPLEXITY',
    'minESSENTIAL_COMPLEXITY', 'minLOC_EXECUTABLE', 'minHALSTEAD_CONTENT', 'minHALSTEAD_DIFFICULTY', 
    'minHALSTEAD_EFFORT', 'minHALSTEAD_ERROR_EST', 'minHALSTEAD_LENGTH', 'minHALSTEAD_LEVEL', 
    'minHALSTEAD_PROG_TIME', 'minHALSTEAD_VOLUME', 'minNUM_OPERANDS', 'minNUM_OPERATORS', 
    'minNUM_UNIQUE_OPERANDS', 'minNUM_UNIQUE_OPERATORS', 'minLOC_TOTAL', 'maxLOC_BLANK', 
    'maxBRANCH_COUNT', 'maxLOC_CODE_AND_COMMENT', 'maxLOC_COMMENTS', 'maxCYCLOMATIC_COMPLEXITY', 
    'maxDESIGN_COMPLEXITY', 'maxESSENTIAL_COMPLEXITY', 'maxLOC_EXECUTABLE', 'maxHALSTEAD_CONTENT', 
    'maxHALSTEAD_DIFFICULTY', 'maxHALSTEAD_EFFORT', 'maxHALSTEAD_ERROR_EST', 'maxHALSTEAD_LENGTH', 
    'maxHALSTEAD_LEVEL', 'maxHALSTEAD_PROG_TIME', 'maxHALSTEAD_VOLUME', 'maxNUM_OPERANDS', 
    'maxNUM_OPERATORS', 'maxNUM_UNIQUE_OPERANDS', 'maxNUM_UNIQUE_OPERATORS', 'maxLOC_TOTAL', 
    'avgLOC_BLANK', 'avgBRANCH_COUNT', 'avgLOC_CODE_AND_COMMENT', 'avgLOC_COMMENTS', 
    'avgCYCLOMATIC_COMPLEXITY', 'avgDESIGN_COMPLEXITY', 'avgESSENTIAL_COMPLEXITY', 'avgLOC_EXECUTABLE', 
    'avgHALSTEAD_CONTENT', 'avgHALSTEAD_DIFFICULTY', 'avgHALSTEAD_EFFORT', 'avgHALSTEAD_ERROR_EST', 
    'avgHALSTEAD_LENGTH', 'avgHALSTEAD_LEVEL', 'avgHALSTEAD_PROG_TIME', 'avgHALSTEAD_VOLUME', 
    'avgNUM_OPERANDS', 'avgNUM_OPERATORS', 'avgNUM_UNIQUE_OPERANDS', 'avgNUM_UNIQUE_OPERATORS', 
    'avgLOC_TOTAL', 'sumLOC_BLANK', 'sumBRANCH_COUNT', 'sumLOC_CODE_AND_COMMENT', 'sumLOC_COMMENTS', 
    'sumCYCLOMATIC_COMPLEXITY', 'sumDESIGN_COMPLEXITY', 'sumESSENTIAL_COMPLEXITY', 'sumLOC_EXECUTABLE', 
    'sumHALSTEAD_CONTENT', 'sumHALSTEAD_DIFFICULTY', 'sumHALSTEAD_EFFORT', 'sumHALSTEAD_ERROR_EST', 
    'sumHALSTEAD_LENGTH', 'sumHALSTEAD_LEVEL', 'sumHALSTEAD_PROG_TIME', 'sumHALSTEAD_VOLUME', 
    'sumNUM_OPERANDS', 'sumNUM_OPERATORS', 'sumNUM_UNIQUE_OPERANDS', 'sumNUM_UNIQUE_OPERATORS', 
    'sumLOC_TOTAL'
]

# Target variable
target = 'DL'

# Step 3: Encode target labels as numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])

# Step 4: Prepare the features and split the data
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Defect", "Defect"], yticklabels=["No Defect", "Defect"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 8: Save the trained XGBoost model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
