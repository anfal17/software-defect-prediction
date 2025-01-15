import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc
)

# Load the pre-trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input ranges based on the dataset
input_ranges = {
    "avgCYCLOMATIC_COMPLEXITY": (1.0, 6.5),
    "NUM_OF_CHILDREN": (0, 4),
    "DEP_ON_CHILD": (0, 2),
    "PERCENT_PUB_DATA": (0, 100),
    "avgLOC_TOTAL": (4.0, 50.3),
    "avgLOC_EXECUTABLE": (0.0, 40.7),
    "COUPLING_BETWEEN_OBJECTS": (4, 24),
    "avgHALSTEAD_EFFORT": (8.375, 14758.77),
}

# Title
st.title("Software Bug Prediction with Detailed Analysis")

st.write("""
This app predicts whether a given piece of code is likely to contain bugs based on software metrics. 
It also provides detailed analysis with a confusion matrix, feature importance, and classification metrics.
""")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
- Adjust sliders or inputs within realistic ranges.
- Hit the **Predict** button for results.
- Explore additional visualizations for detailed insights.
""")

# Inputs for each metric
st.header("Input Software Metrics")

inputs = {}
for metric, (min_val, max_val) in input_ranges.items():
    if isinstance(min_val, int) and isinstance(max_val, int):
        inputs[metric] = st.slider(
            metric, min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2, step=1
        )
    elif metric == "avgHALSTEAD_EFFORT":
        inputs[metric] = st.number_input(
            metric, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=1.0
        )
    else:
        inputs[metric] = st.slider(
            metric, min_value=float(min_val), max_value=float(max_val), value=(min_val + max_val) / 2
        )

# Prediction button
if st.button("Predict"):
    try:
        # Convert inputs to a DataFrame
        input_df = pd.DataFrame([inputs])

        # Scale the input data
        scaled_data = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data) if hasattr(model, "predict_proba") else None

        # Display prediction
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The model predicts that the code is likely to have a bug!")
        else:
            st.success("The model predicts that the code is unlikely to have a bug.")

        # if prediction_proba is not None:
        #     st.write("Prediction Probabilities:")
        #     st.write(f"Probability of No Bug: {prediction_proba[0][0]:.2f}")
        #     st.write(f"Probability of Bug: {prediction_proba[0][1]:.2f}")

        # Show input metrics
        st.subheader("Input Metrics")
        st.write(input_df)

        # Visualizations and analysis
        # st.header("Model Analysis")

        # # Confusion Matrix
        # st.subheader("Confusion Matrix")
        # y_test = [1, 0, 1, 1, 0]  # Replace with actual y_test during evaluation
        # y_pred = [1, 0, 1, 0, 1]  # Replace with model predictions on test data
        # cm = confusion_matrix(y_test, y_pred)
        # fig, ax = plt.subplots()
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Bug", "Bug"], yticklabels=["No Bug", "Bug"])
        # plt.ylabel("Actual")
        # plt.xlabel("Predicted")
        # st.pyplot(fig)

        # # Classification Report
        # st.subheader("Classification Report")
        # report = classification_report(y_test, y_pred, target_names=["No Bug", "Bug"], output_dict=True)
        # st.dataframe(pd.DataFrame(report).transpose())

        # Visualizations and analysis
        st.header("Model Analysis")

        # Load test data from the CSV file
        test_data = pd.read_csv('test_data.csv')  # Load the test data CSV file

        # Separate features and target variable from test data
        X_test = test_data.drop(columns=['DL'])  # Replace 'DL' with the actual target column name if it's different
        y_test = test_data['DL']

        # Scale the test features using the same scaler used during training
        X_test_scaled = scaler.transform(X_test)

        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Bug", "Bug"], yticklabels=["No Bug", "Bug"])
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(fig)

        from sklearn.metrics import accuracy_score, precision_score, recall_score

        # Accuracy, Precision, and Recall for both classes (No Bug and Bug)
        st.subheader("Performance Metrics")

        # Calculate accuracy, precision, and recall for "No Bug" and "Bug"
        accuracy = accuracy_score(y_test, y_pred)
        precision_bug = precision_score(y_test, y_pred, pos_label=1)  # Precision for Bug
        recall_bug = recall_score(y_test, y_pred, pos_label=1)  # Recall for Bug
        precision_no_bug = precision_score(y_test, y_pred, pos_label=0)  # Precision for No Bug
        recall_no_bug = recall_score(y_test, y_pred, pos_label=0)  # Recall for No Bug

        # Display the results
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision (Bug): {precision_bug:.2f}")
        st.write(f"Recall (Bug): {recall_bug:.2f}")
        st.write(f"Precision (No Bug): {precision_no_bug:.2f}")
        st.write(f"Recall (No Bug): {recall_no_bug:.2f}")


        # # ROC Curve
        # st.subheader("ROC Curve")
        # y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7])  # Replace with model decision function/probabilities
        # fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        # roc_auc = auc(fpr, tpr)
        # fig_roc, ax_roc = plt.subplots()
        # ax_roc.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
        # ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        # ax_roc.set_xlabel("False Positive Rate")
        # ax_roc.set_ylabel("True Positive Rate")
        # ax_roc.legend(loc="lower right")
        # st.pyplot(fig_roc)

        # # Feature Importance
        # st.subheader("Feature Importance")
        # if hasattr(model, "feature_importances_"):
        #     feature_importances = model.feature_importances_
        #     importance_df = pd.DataFrame(
        #         {"Feature": input_df.columns, "Importance": feature_importances}
        #     ).sort_values(by="Importance", ascending=False)
        #     st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.error(f"Error during prediction or visualization: {e}")
else:
    st.info("Adjust the sliders and click **Predict** to get results.")
