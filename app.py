import streamlit as st
import pandas as pd
import joblib

# Set up page configuration
st.title("Drive Failure Prediction Tool")
st.write("Enter the hard drive SMART metrics below to predict potential failure.")

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()

    # Create user input fields
    smart_5 = st.number_input("SMART 5 (Reallocated Sectors Count)", min_value=0, value=0)
    smart_187 = st.number_input("SMART 187 (Reported Uncorrectable)", min_value=0, value=0)
    smart_197 = st.number_input("SMART 197 (Current Pending Sector)", min_value=0, value=0)

    # Prediction button
    if st.button("Predict Drive Health"):
        input_data = pd.DataFrame(
            [[smart_5, smart_187, smart_197]], 
            columns=['smart_5_raw', 'smart_187_raw', 'smart_197_raw']
        )
        
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("High Risk of Failure Detected")
        else:
            st.success("Drive Status: Healthy")

except FileNotFoundError:
    st.warning("`model.pkl` not found. Please upload your trained model file to the repository.")