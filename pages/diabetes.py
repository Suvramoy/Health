import config as cfg
import joblib
import numpy as np
import streamlit as st
from config import doctor_search


def predict_early_diabetes(
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age
):
    model = joblib.load(r'models\Diabetes_pipeline_rf.pkl')
    prediction = model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]]))
    return prediction


def diabetes_app():
    st.set_page_config(
        page_title="Healthify - Diabetes Diagnosis",
        page_icon="🏥",
    )
    st.markdown(
        f"<h1 style='text-align: center; color: black;'>Early Diabetes Diagnosis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Choose below options according to the report to know the patient's status</h4>",
        unsafe_allow_html=True,
    )

    Pregnancies = st.slider("Pregnancie ", min_value=0, max_value=17)
    Glucose = st.slider("Glucose ", min_value=50, max_value=200)
    BloodPressure = st.slider("BloodPressure ", min_value=5, max_value=200)
    SkinThickness = st.slider("SkinThickness ", min_value=5, max_value=100)
    Insulin = st.slider("Insulin ", min_value=5, max_value=800)
    BMI = st.slider("BMI ", min_value=10, max_value=60)
    DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction ", min_value=0, max_value=3)
    Age = st.slider("Age ", min_value=21, max_value=70)


    result = None
    if st.button("Predict"):
        result = predict_early_diabetes(
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age
        )
        if result == 1:
            st.subheader("The patient have high chances of having Diabetes 😔")

            st.markdown("---")
            st.error(
                "If you are a patient, consult with one of the following doctors immediately"
            )
            st.subheader("Specialists 👨‍⚕")

            st.write(
                "Click on the specialists to get the specialists nearest to your location 📍"
            )
            pcp = doctor_search("Primary Care Provider")
            infec = doctor_search("Endocrinologist")
            st.markdown(f"- [Primary Care Doctor]({pcp}) 👨‍⚕")
            st.markdown(f"- [Endocrinologist]({infec}) 👨‍⚕")
            st.markdown("---")
        else:
            st.subheader("The patient does not have Diabetes 😄")


if __name__ == "__main__":
    diabetes_app()