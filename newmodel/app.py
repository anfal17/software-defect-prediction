import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the saved model and scaler
model = joblib.load('best_model.pkl')  # Load the best model
scaler = joblib.load('scaler.pkl')     # Load the scaler

# Function to predict based on user input
def predict(features):
    features_scaled = scaler.transform([features])  # Scale the features
    prediction = model.predict(features_scaled)
    return prediction

# Streamlit user interface
st.title("Software Defect Prediction")

st.write("""
    This model predicts whether the software is likely **defective** or **not defective** based on various software attributes.
    The metrics provided below help evaluate the model's performance.
""")

# Input Section
st.header("Input Parameters")

# Cyclomatic Complexity
st.write("Cyclomatic Complexity: Measures the complexity of the program's control flow.")
avgCYCLOMATIC_COMPLEXITY = st.slider("Avg Cyclomatic Complexity", 0, 100, 20)

# Number of Children
st.write("Number of Children: Represents the number of modules/functions a particular function calls.")
NUM_OF_CHILDREN = st.slider("Num of Children", 0, 50, 5)

# Dependence on Child
st.write("Dependence on Child: Measures the dependency of a function on child modules.")
DEP_ON_CHILD = st.slider("Dependence on Child", 0, 100, 10)

# Percent of Public Data
st.write("Percentage of Public Data: Fraction of code made publicly available.")
PERCENT_PUB_DATA = st.slider("Percent Public Data", 0.0, 100.0, 50.0)

# LOC Total (Lines of Code)
st.write("Lines of Code: Measures the size of the software.")
avgLOC_TOTAL = st.slider("Avg LOC Total", 0, 10000, 500)

# LOC Executable
st.write("Executable Lines of Code: Measures how much of the code is executable.")
avgLOC_EXECUTABLE = st.slider("Avg LOC Executable", 0, 10000, 100)

# Coupling Between Objects
st.write("Coupling Between Objects: Measures the interdependence between objects in the system.")
COUPLING_BETWEEN_OBJECTS = st.slider("Coupling Between Objects", 0, 100, 30)

# Halstead Effort
st.write("Halstead Effort: Measures the effort required to understand the code based on Halstead metrics.")
avgHALSTEAD_EFFORT = st.slider("Avg Halstead Effort", 0, 1000, 100)

# Button to show Java code
if st.button('Show Sample Java Code'):
    st.write("""
        Here's a simple Java code that checks for potential defects by analyzing the cyclomatic complexity and lines of code:
        
        ```java
        public class SoftwareDefect {
            public static void main(String[] args) {
                int cyclomaticComplexity = 25;  // Example metric
                int linesOfCode = 2000;  // Example metric
                int thresholdComplexity = 15;
                int thresholdLOC = 1500;
                
                if(cyclomaticComplexity > thresholdComplexity || linesOfCode > thresholdLOC) {
                    System.out.println("Potential software defect detected!");
                } else {
                    System.out.println("Software seems stable.");
                }
            }
        }
        ```
    """)
    st.write("In the above code, if the cyclomatic complexity or lines of code exceed the threshold, the software is flagged as potentially defective.")

# Button to trigger prediction
if st.button('Predict'):
    features = [avgCYCLOMATIC_COMPLEXITY, NUM_OF_CHILDREN, DEP_ON_CHILD, PERCENT_PUB_DATA,
                avgLOC_TOTAL, avgLOC_EXECUTABLE, COUPLING_BETWEEN_OBJECTS, avgHALSTEAD_EFFORT]
    prediction = predict(features)
    
    if prediction == 1:
        st.write("The software is likely **defective**.")
    else:
        st.write("The software is likely **not defective**.")
    
    # Evaluate model metrics (use actual test data for real-world use)
    # For demonstration, let's generate some fake predictions and compare them
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Example true labels
    y_pred = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]  # Example predicted labels

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Show other metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1-Score**: {f1:.2f}")

    # Display additional metrics in a bar plot
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color='green')
    ax.set_title('Model Performance Metrics')
    ax.set_ylabel('Score')
    st.pyplot(fig)

# Model Evaluation Metrics Explanation
st.header("Model Evaluation Metrics")

st.write("""
    The following metrics are used to evaluate the model's performance on test data:
    - **Accuracy**: Proportion of correct predictions made by the model.
    - **Precision**: Measures the accuracy of positive predictions (how many of the predicted positives are actually positive).
    - **Recall**: Measures the ability to find all relevant positive instances.
    - **F1-Score**: Harmonic mean of Precision and Recall, providing a balance between the two.
""")

# Sidebar for Additional Information
st.sidebar.header("Additional Information")

st.sidebar.write("""
    - The model takes into account various code quality metrics to predict defects.
    - Lower values in Cyclomatic Complexity and LOC (Lines of Code) might suggest more maintainable code, potentially leading to fewer defects.
    - A higher value of "Coupling Between Objects" could indicate greater complexity and interdependencies, increasing the likelihood of defects.
""")
