import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the trained XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature descriptions and example values
features_info = {
    'COUPLING_BETWEEN_OBJECTS': ('The number of connections a class has with other classes.', 0, 10),
    'LACK_OF_COHESION_OF_METHODS': ('Measures how related the methods in a class are.', 0, 1),
    'NUM_OF_CHILDREN': ('The number of subclasses a class has.', 0, 20),
    'FAN_IN': ('The number of methods that call this class.', 0, 15),
    'RESPONSE_FOR_CLASS': ('The number of methods executed in response to a message.', 1, 25),
    'avgCYCLOMATIC_COMPLEXITY': ('Average number of independent paths in the code.', 1, 10),
    'sumHALSTEAD_DIFFICULTY': ('Summation of effort required to understand the code.', 10, 100),
    'maxLOC_EXECUTABLE': ('Maximum number of executable lines of code in a method.', 10, 300)
}

# Initialize past predictions
if "past_predictions" not in st.session_state:
    st.session_state.past_predictions = []

# App layout
st.title("Defect Prediction Model")
col1, col2 = st.columns(2)

# Left column: Input sliders
with col1:
    st.header("Input Features")
    input_data = {}
    for feature, (desc, min_val, max_val) in features_info.items():
        st.markdown(f"**{feature}**: {desc}")
        input_data[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2)

    # Convert input into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"The predicted class is: {'Defective' if prediction == 1 else 'Non-Defective'}")

        # Append to past predictions
        st.session_state.past_predictions.append((input_data, 'Defective' if prediction == 1 else 'Non-Defective'))

    # Past predictions timeline
    st.subheader("Past Predictions")
    for i, (features, prediction) in enumerate(st.session_state.past_predictions):
        st.markdown(f"**Prediction {i+1}:** {prediction}")
        st.write(features)

# Right column: Java example
with col2:
    st.header("Java Example for Metrics")
    java_code = """
// Java Example: Understanding Metrics

public class MetricsExample {
    public static void main(String[] args) {
        // Number of connections a class has with others
        int couplingBetweenObjects = 5;

        // Measure of how related methods in a class are
        double lackOfCohesionOfMethods = 0.6;

        // Number of subclasses a class has
        int numOfChildren = 3;

        // Number of methods that call this class
        int fanIn = 7;

        // Number of methods executed in response to a message
        int responseForClass = 12;

        // Average number of independent paths in the code
        double avgCyclomaticComplexity = 4.5;

        // Sum of effort required to understand the code
        double sumHalsteadDifficulty = 45.0;

        // Maximum number of executable lines of code in a method
        int maxLocExecutable = 120;

        // Print these metrics
        System.out.println("Coupling Between Objects: " + couplingBetweenObjects);
        System.out.println("Lack of Cohesion of Methods: " + lackOfCohesionOfMethods);
        System.out.println("Number of Children: " + numOfChildren);
        System.out.println("Fan-In: " + fanIn);
        System.out.println("Response for Class: " + responseForClass);
        System.out.println("Average Cyclomatic Complexity: " + avgCyclomaticComplexity);
        System.out.println("Sum of Halstead Difficulty: " + sumHalsteadDifficulty);
        System.out.println("Max LOC Executable: " + maxLocExecutable);
    }
}
"""
    st.code(java_code, language="java")

    st.markdown("**Explanation**: The Java code above helps visualize the meaning of each metric with example values. It doesn't predict but demonstrates where each feature applies in real scenarios.")

# Confusion matrix
st.header("Confusion Matrix")
true_labels = [0, 1, 0, 1]  # Dummy data for visualization
predicted_labels = [0, 1, 0, 1]
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Defect", "Defect"], yticklabels=["No Defect", "Defect"])
st.pyplot(plt)
