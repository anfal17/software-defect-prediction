import pandas as pd

# Load the original CSV file
data = pd.read_csv('kc1-class-level-defectiveornot.csv')

# List of relevant columns (input features and target)
columns_of_interest = [
    'CYCLOMATIC_COMPLEXITY',
    'DESIGN_COMPLEXITY',
    'HALSTEAD_DIFFICULTY',
    'HALSTEAD_EFFORT',
    'HALSTEAD_ERROR_EST',
    'HALSTEAD_LENGTH',
    'HALSTEAD_LEVEL',
    'HALSTEAD_PROG_TIME',
    'HALSTEAD_VOLUME',
    'LOC_BLANK',
    'LOC_CODE_AND_COMMENT',
    'LOC_COMMENTS',
    'BRANCH_COUNT',
    'ESSENTIAL_COMPLEXITY',
    'NUM_OPERANDS',
    'NUM_OPERATORS',
    'NUM_UNIQUE_OPERANDS',
    'NUM_UNIQUE_OPERATORS',
    'LOC_EXECUTABLE',
    'LOC_TOTAL',
    'DL'
]

# Extract only the relevant columns
extracted_data = data[columns_of_interest]

# Check for missing values (optional - you can handle missing values here)
# For example, replace NaN values with the column mean:
extracted_data = extracted_data.fillna(extracted_data.mean())

# Save the new CSV file with only the selected columns
extracted_data.to_csv('extracted_features.csv', index=False)

print("New CSV with extracted features has been created: 'extracted_features.csv'")
