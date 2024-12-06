import streamlit as st
import joblib
import numpy as np
import ast
from radon.complexity import cc_visit
from radon.raw import analyze

# Load the saved model
model = joblib.load('xgb_model.pkl')

# Function to extract code metrics
def extract_code_metrics(code):
    # Get raw metrics (lines of code, blank lines, etc.)
    raw_metrics = analyze(code)

    # Cyclomatic Complexity for functions and methods
    cc = cc_visit(code)  # Cyclomatic complexity for all functions/methods in the code

    # Calculate Maintainability Index (MI) based on Cyclomatic Complexity, Halstead Volume, and LOC
    ave_cc = np.mean([x.complexity for x in cc]) if cc else 0
    halstead_volume = raw_metrics.loc  # Correct attribute access
    loc = raw_metrics.loc  # Correct attribute access

    # Maintainability Index formula
    if loc > 0:
        mi = 171 - 5.2 * np.log(ave_cc + 1) - 0.23 * halstead_volume - 16.2 * np.log(loc)
    else:
        mi = 0  # Handle edge case where LOC is zero

    # Analyze code with AST to count number of classes and methods
    tree = ast.parse(code)
    noc = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))  # Count number of classes
    methods = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))  # Count number of methods
    attributes = sum(isinstance(node, ast.Assign) for node in ast.walk(tree))  # Count number of attributes

    # Counting operands and operators using AST
    num_operands = 0
    num_operators = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):  # Binary operations
            num_operands += 2  # A binary operation has two operands (left and right)
            num_operators += 1  # A binary operation has one operator
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            num_operands += 1  # A unary operation has one operand
            num_operators += 1  # A unary operation has one operator
        elif isinstance(node, ast.Call):  # Function calls
            num_operands += len(node.args)  # Arguments as operands
            num_operators += 1  # Function call as an operator

    # Halstead Metrics (content, difficulty, effort, etc.) - placeholders
    halstead_content = raw_metrics.loc  # Placeholder for Halstead content
    halstead_difficulty = 1  # Placeholder for Halstead difficulty
    halstead_effort = 1  # Placeholder for Halstead effort
    halstead_error_est = 1  # Placeholder for Halstead error estimate
    halstead_length = raw_metrics.loc  # Placeholder for Halstead length
    halstead_level = 1  # Placeholder for Halstead level
    halstead_prog_time = 1  # Placeholder for Halstead program time
    halstead_volume = 1  # Placeholder for Halstead volume

    # Metrics dictionary
    metrics_dict = {
        'LOC_BLANK': raw_metrics.blank,  # Correct attribute access
        'BRANCH_COUNT': len(cc),  # Branch count as number of decision points
        'LOC_CODE_AND_COMMENT': raw_metrics.loc,  # Correct attribute access
        'LOC_COMMENTS': raw_metrics.comments,  # Correct attribute access
        'CYCLOMATIC_COMPLEXITY': sum([x.complexity for x in cc]),  # Sum of complexities
        'DESIGN_COMPLEXITY': len(cc),  # Placeholder for design complexity
        'ESSENTIAL_COMPLEXITY': len(cc),  # Placeholder for essential complexity
        'LOC_EXECUTABLE': raw_metrics.loc,  # Correct attribute access
        'HALSTEAD_CONTENT': halstead_content,  # Halstead content
        'HALSTEAD_DIFFICULTY': halstead_difficulty,  # Halstead difficulty
        'HALSTEAD_EFFORT': halstead_effort,  # Halstead effort
        'HALSTEAD_ERROR_EST': halstead_error_est,  # Halstead error estimate
        'HALSTEAD_LENGTH': halstead_length,  # Halstead length
        'HALSTEAD_LEVEL': halstead_level,  # Halstead level
        'HALSTEAD_PROG_TIME': halstead_prog_time,  # Halstead program time
        'HALSTEAD_VOLUME': halstead_volume,  # Halstead volume
        'NUM_OPERANDS': num_operands,  # Total number of operands
        'NUM_OPERATORS': num_operators,  # Total number of operators
        'NUM_UNIQUE_OPERANDS': len(set([x for x in ast.walk(tree) if isinstance(x, ast.Name)])),  # Unique operands
        'NUM_UNIQUE_OPERATORS': len(set([x for x in ast.walk(tree) if isinstance(x, (ast.BinOp, ast.UnaryOp))])),  # Unique operators
        'LOC_TOTAL': raw_metrics.loc,  # Correct attribute access
        'MAINTAINABILITY_INDEX': mi,  # Maintainability Index
        'NUM_CLASSES': noc,  # Number of classes
        'NUM_METHODS': methods,  # Number of methods
        'NUM_ATTRIBUTES': attributes,  # Number of attributes
    }

    return metrics_dict

# Function to preprocess the metrics dictionary
def preprocess_metrics(metrics_dict):
    # Only use the 21 features that the model was trained on
    feature_columns = [
        'LOC_BLANK', 'BRANCH_COUNT', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'CYCLOMATIC_COMPLEXITY', 
        'DESIGN_COMPLEXITY', 'ESSENTIAL_COMPLEXITY', 'LOC_EXECUTABLE', 'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 
        'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 
        'HALSTEAD_VOLUME', 'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS', 
        'LOC_TOTAL', 'MAINTAINABILITY_INDEX', 'NUM_CLASSES', 'NUM_METHODS', 'NUM_ATTRIBUTES'
    ]

    # Make sure the dictionary has only the required columns (21 features)
    if len(metrics_dict) != len(feature_columns):
        raise ValueError(f"Feature mismatch: expected {len(feature_columns)} features, got {len(metrics_dict)}")

    # Prepare the feature array
    features = [metrics_dict.get(col, 0) for col in feature_columns]
    return np.array(features).reshape(1, -1)

# Function to predict bugs based on code
def predict_bugs(code):
    # Extract metrics from code
    metrics_dict = extract_code_metrics(code)

    # Preprocess the extracted metrics
    features = preprocess_metrics(metrics_dict)

    # Predict with the trained model
    prediction = model.predict(features)
    if prediction == 0:
        return "No bugs predicted"
    else:
        return "Bugs predicted"

# Streamlit UI
st.title("Software Bug Prediction")
st.subheader("Enter your code below:")

# Input code box for user
user_code = st.text_area("Enter Code", height=200)

if st.button("Predict Bugs"):
    if user_code.strip() != "":
        result = predict_bugs(user_code)
        st.success(result)
    else:
        st.error("Please enter some code to analyze.")
