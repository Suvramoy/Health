import numpy as np
import pandas as pd
import streamlit as st
import joblib
from config import doctor_search
import sklearn

# Load the model
model = joblib.load(r'models\Heart_pipeline_gb.pkl')



# Define the prediction function
def prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Convert input strings to integers (if necessary)
    sex = int(sex)
    cp = int(cp)
    fbs = int(fbs)
    restecg = int(restecg)
    exang = int(exang)
    slope = int(slope)
    ca = int(ca)
    thal = int(thal)

    # Create a DataFrame with the same columns used during training
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    )
    
    # Ensure that the data types match the expected types in the model
    numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    input_data[numerical_columns] = input_data[numerical_columns].astype(float)
    input_data[categorical_columns] = input_data[categorical_columns].astype(object)

    # Check for any NaN values in the input data
    if input_data.isnull().values.any():
        raise ValueError("Input contains NaN values. Please provide valid inputs.")

    # Perform prediction
    prediction = model.predict(input_data)
    return prediction


def main():
    st.set_page_config(page_title="Heart", page_icon="🏥")

    # User inputs via Streamlit components
    age = st.slider('Age in years', min_value=25, max_value=80)
    sex = st.radio("Gender (1 = male; 0 = female)", ('0', '1'))
    cp = st.selectbox("Chest pain type", ['0', '1', '2', '3'])
    trestbps = st.slider('Resting blood pressure (in mm Hg on admission to the hospital)', min_value=90, max_value=200)
    chol = st.slider('Serum cholestoral in mg/dl', min_value=120, max_value=570)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", ('0', '1'))
    restecg = st.radio("Resting electrocardiographic results", ('0', '1', '2'))
    thalach = st.slider('Maximum heart rate achieved', min_value=70, max_value=200)
    exang = st.radio("Exercise induced angina (1 = yes; 0 = no)", ('0', '1'))
    oldpeak = st.slider('ST depression induced by exercise relative to rest', min_value=0.0, max_value=7.0, step=0.1)
    slope = st.radio("The slope of the peak exercise ST segment", ('0', '1', '2'))
    ca = st.radio("Number of major vessels (0-4) colored by flourosopy", ('0', '1', '2', '3', '4'))
    thal = st.radio("1 = normal; 2 = fixed defect; 3 = reversible defect", ('1', '2', '3'))

    result = None
    if st.button("Predict"):
        # Call the prediction function with the user's input
        result = prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        
        # Display the result
        if result[0] == 1:  # Assuming the model returns a list/array
            st.subheader("The patient has high chances of having heart disease 😔")
            st.markdown("---")
            st.error("If you are a patient, consult with one of the following doctors immediately")
            st.subheader("Specialists 👨‍⚕")
            st.write("Click on the specialists to get the specialists nearest to your location 📍")
            pcp = doctor_search("Primary Care Provider")
            cardiologist = doctor_search("Cardiologist")
            st.markdown(f"- [Primary Care Doctor]({pcp}) 👨‍⚕")
            st.markdown(f"- [Cardiologist]({cardiologist}) 👨‍⚕")
            st.markdown("---")
        else:
            st.subheader("The patient does not have heart disease 😄")

if __name__ == '__main__':
    main()
