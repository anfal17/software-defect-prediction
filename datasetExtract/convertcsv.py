import pandas as pd

# Load the original CSV file
data = pd.read_csv('kc1-class-level-defectiveornot.csv')

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

# Check if all specified columns exist in the DataFrame
missing_columns = [col for col in columns_of_interest if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the data: {missing_columns}")

# Extract only the relevant columns
extracted_data = data[columns_of_interest]

# Convert boolean-like string values ('_TRUE', 'TRUE', 'FALSE') to numeric (1, 0) if present
boolean_columns = ['DL']  # Replace with actual columns containing boolean-like strings
for col in boolean_columns:
    if extracted_data[col].dtype == object:  # Likely contains strings
        # Handle '_TRUE', 'TRUE', 'FALSE' and map them to numeric values (1 for TRUE, 0 for FALSE)
        extracted_data[col] = extracted_data[col].replace({'_TRUE': 1, 'TRUE': 1, 'FALSE': 0})

# Handle missing values by replacing them with column means (optional)
extracted_data = extracted_data.fillna(extracted_data.mean(numeric_only=True))

# Save the new CSV file with only the selected columns
extracted_data.to_csv('extracted_features.csv', index=False)

print("New CSV with extracted features has been created: 'extracted_features.csv'")
