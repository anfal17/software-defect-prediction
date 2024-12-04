import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load the trained model and scaler
model = joblib.load('bug_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize an empty list to store prediction history (this could be stored in a database for persistence)
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Define Streamlit user interface
st.title("Software Bug Prediction")

# Input fields
st.subheader("Model Input:")
LOC = st.number_input("Lines of Code (LOC)", min_value=0)
complexity = st.number_input("Cyclomatic Complexity", min_value=0)
methods = st.number_input("Methods Count", min_value=0)

# Prepare the input for prediction
input_data = np.array([[LOC, complexity, methods]])

# Button to trigger prediction
if st.button("Predict"):
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_scaled)
    
    # Display prediction
    if prediction == 1:
        st.write("This code is likely to be buggy.")
    else:
        st.write("This code is likely to be bug-free.")
    
    # Save prediction history for graphing and future reference
    st.session_state.prediction_history.append((LOC, complexity, methods, prediction[0]))

    # Display prediction history (display top 10 latest predictions)
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history, columns=['LOC', 'Complexity', 'Methods', 'Prediction'])
    st.write(history_df.tail(10))  # Show the last 10 predictions

    # For confusion matrix, let's simulate some predictions using random data
    # Generate some random test data for confusion matrix (this would ideally be the actual test set in a real scenario)
    simulated_y_test = np.random.choice([0, 1], size=(10,))  # Simulated true labels (buggy or not)
    simulated_X_test = np.random.randint(1, 100, size=(10, 3))  # Simulated input data (3 features: LOC, complexity, methods)

    # Scale the simulated test data
    simulated_X_test_scaled = scaler.transform(simulated_X_test)

    # Make predictions on the simulated test data
    simulated_y_pred = model.predict(simulated_X_test_scaled)

    # Generate the confusion matrix
    cm = confusion_matrix(simulated_y_test, simulated_y_pred)
    st.subheader("Confusion Matrix")
    st.write(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    plt.xticks(np.arange(2), ['Not Buggy', 'Buggy'])
    plt.yticks(np.arange(2), ['Not Buggy', 'Buggy'])
    st.pyplot(fig)

# Optional: You can add a chart to visualize the history of bug predictions (or any other analytics you want)
if len(st.session_state.prediction_history) > 0:
    st.subheader("Prediction History Chart")
    history_df = pd.DataFrame(st.session_state.prediction_history, columns=['LOC', 'Complexity', 'Methods', 'Prediction'])
    
    # Plot the number of buggy predictions over time
    history_df['Prediction'] = history_df['Prediction'].map({0: 'Bug-Free', 1: 'Buggy'})
    prediction_counts = history_df['Prediction'].value_counts()
    
    # Display bar chart
    st.bar_chart(prediction_counts)
