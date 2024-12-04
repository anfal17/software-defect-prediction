import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the pre-trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Display the page in a wide format
st.set_page_config(layout="wide")

# Create a two-column layout
col1, col2 = st.columns([1, 1])  # Adjust column width ratio as needed

# Column 1: Interactive Sliders and Input
with col1:
    st.title("Defect Prediction Model")
    st.write("Use the sliders and input boxes below to adjust feature values and make predictions.")

    # Example features
    percent_pub_data = st.slider("Percent Public Data (%)", 0, 100, 50, help="Example: 25 means 25% public data.")
    access_to_pub_data = st.slider("Access to Public Data (Count)", 0, 50, 10, help="Example: Number of public methods used.")
    coupling_between_objects = st.slider("Coupling Between Objects (CBO)", 0, 50, 5, help="Example: 10 indicates high coupling.")
    depth = st.slider("Depth of Inheritance Tree (Depth)", 0, 10, 2, help="Example: Depth of 3 indicates deeper inheritance.")
    fan_in = st.slider("Fan In", 0, 100, 20, help="Example: 30 means this class is called by 30 other classes.")
    lack_of_cohesion = st.slider("Lack of Cohesion (LCOM)", 0.0, 1.0, 0.5, help="Example: 0.5 indicates moderate cohesion.")
    cyclomatic_complexity = st.slider("Cyclomatic Complexity", 0, 20, 5, help="Example: 5 indicates moderate branching.")
    halstead_effort = st.number_input("Halstead Effort", min_value=0.0, value=50.0, help="Example: 50 represents code comprehension effort.")

    # Prepare input data for prediction
    input_data = pd.DataFrame([[
        percent_pub_data, access_to_pub_data, coupling_between_objects, depth,
        fan_in, lack_of_cohesion, cyclomatic_complexity, halstead_effort
    ]], columns=[
        'PERCENT_PUB_DATA', 'ACCESS_TO_PUB_DATA', 'COUPLING_BETWEEN_OBJECTS', 'DEPTH',
        'FAN_IN', 'LACK_OF_COHESION_OF_METHODS', 'CYCLOMATIC_COMPLEXITY', 'HALSTEAD_EFFORT'
    ])

    # Scale the input data
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data)

    # Make a prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_input)
        st.success(f"Prediction: {'Defective' if prediction[0] == 1 else 'Not Defective'}")

# Column 2: Java Code Example
with col2:
    st.title("Metrics in Java Code")
    st.write("Here's a Java code example to demonstrate the metrics:")

    java_code = """
    public class Library {
        private String libraryName;
        private int numBooks;

        public Library(String name) {
            this.libraryName = name;
            this.numBooks = 0;
        }

        public void addBooks(int count) {
            this.numBooks += count;
        }

        public void removeBooks(int count) {
            if (count <= this.numBooks) {
                this.numBooks -= count;
            } else {
                System.out.println("Not enough books to remove!");
            }
        }

        public int getNumBooks() {
            return this.numBooks;
        }

        public void printDetails() {
            System.out.println("Library Name: " + this.libraryName);
            System.out.println("Number of Books: " + this.numBooks);
        }
    }
    """
    # Add a scrollable box for the Java code
    st.code(java_code, language="java")
    st.write("""
    **Metrics Explanation for This Code:**
    - **CBO:** Coupling Between Objects; interacts with the `String` class (Value: 1).
    - **LCOM:** Lack of Cohesion of Methods; moderate cohesion as methods interact with class fields (Value: 0.5).
    - **Cyclomatic Complexity:** Measures decision points; `removeBooks` has an `if-else` condition (Value: 2).
    """)

# Add a footer to separate past predictions
st.markdown("---")
st.subheader("Past Predictions Timeline")
st.write("Feature values and their corresponding predictions will be shown here for tracking.")
