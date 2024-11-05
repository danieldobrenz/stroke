import streamlit as st
import pickle
import pandas as pd

# Use Markdown with HTML for centered title
st.markdown("<h1 style='text-align: center;'>Stroke Prediction</h1>", unsafe_allow_html=True)

# Display image from a local file
st.image("stroke.jpeg")

st.write("""
Stroke is the leading cause of disability worldwide and the second leading cause of death. The Global Stroke Factsheet released in 2022 reveals that lifetime risk of developing a stroke has increased by 50% over the last 17 years and now 1 in 4 people is estimated to have a stroke in their lifetime. From 1990 to 2019, there has been a 70% increase in stroke incidence, 43% increase in deaths due to stroke, 102% increase in stroke prevalence and 143% increase in Disability Adjusted Life Years (DALY). The most striking feature is that the bulk of the global stroke burden (86% of deaths due to stroke and 89% of DALYs) occur in lower and lower-middle-income countries. This disproportionate burden experienced by lower and lower-middle income countries has posed an unprecedented problem to families with less resources.
""")

st.write("""
The purpose of the project is to predict whether or not someone will have a stroke. The goal is to predict stroke occurrence with high precision (the ratio of true positive predictions to the total predicted positives). We want to minimize the false positives, which incorrectly predict stroke. This is important when treatments could be costly or dangerous to the patient.
""")

st.write(""" Fill out the various items below to see whether or not you are likely to have a stroke: """)

# Gender
gender = st.selectbox('Select Gender:', ('Select', 'Male', 'Female'))
gender_male = 1 if gender == 'Male' else 0
gender_female = 1 if gender == 'Female' else 0

# Age
age = st.text_input('Enter Age:')

# Hypertension
hypertension = st.selectbox('Hypertension (High Blood Pressure):', ['Select', 'Yes', 'No'])
binary_input_hypertension = 1 if hypertension == "Yes" else 0

# Heart Disease
heart_disease = st.selectbox('Heart Disease:', ['Select', 'Yes', 'No'])
binary_input_heart_disease = 1 if heart_disease == "Yes" else 0

# Marital Status
married = st.selectbox('Married:', ('Select', 'Yes', 'No'))
ever_married_yes = 1 if married == "Yes" else 0
ever_married_no  = 1 if married == "No" else 0

# Work Type
worktype = st.selectbox('Work Type:', ('Select', 'Children', 'Government Job', 'Never Worked', 'Private', 'Self-Employed'))
work_type_govt_job = 1 if worktype == "Government Job" else 0
work_type_never_worked = 1 if worktype == "Never Worked" else 0
work_type_private = 1 if worktype == 'Private' else 0
work_type_self_employed = 1 if worktype == 'Self-Employed' else 0
work_type_children = 1 if worktype == 'Children' else 0

# Residence Type
residence_type = st.selectbox('Type of Residence:', ('Select', 'Urban', 'Rural'))
residence_type_rural = 1 if residence_type == "Rural" else 0
residence_type_urban = 1 if residence_type == "Urban" else 0

# Glucose Level
glucose_level = st.text_input('Enter Average Glucose Level:')

# BMI
bmi = st.text_input('Enter BMI (Body Mass Index):')

# Smoking Status
smoker = st.selectbox('Smoking Status/History:', ('Select', 'Use to Smoke', 'Never', 'Smokes', 'Unknown'))
smoking_status_unknown = 1 if smoker == 'Unknown' else 0
smoking_status_formerly_smoked = 1 if smoker == 'Use to Smoke' else 0
smoking_status_never_smoked = 1 if smoker == 'Never' else 0
smoking_status_smokes = 1 if smoker == 'Smokes' else 0

# Load the model
@st.cache_resource
def load_model():
    with open('stroke_random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model()

# Prepare the input data for prediction
input_df = pd.DataFrame({
    'gender_male': [gender_male],
    'gender_female': [gender_female],
    'age': [age],
    'hypertension': [binary_input_hypertension],
    'heart_disease': [binary_input_heart_disease],
    'ever_married_yes': [ever_married_yes],
    'ever_married_no': [ever_married_no],
    'work_type_govt_job': [work_type_govt_job],
    'work_type_never_worked': [work_type_never_worked],
    'work_type_private': [work_type_private],
    'work_type_self_employed': [work_type_self_employed],
    'work_type_children': [work_type_children],
    'residence_type_rural': [residence_type_rural],
    'residence_type_urban': [residence_type_urban],
    'avg_glucose_level': [glucose_level],
    'bmi': [bmi],
    'smoking_status_unknown': [smoking_status_unknown],
    'smoking_status_formerly_smoked': [smoking_status_formerly_smoked],
    'smoking_status_never_smoked': [smoking_status_never_smoked],
    'smoking_status_smokes': [smoking_status_smokes]
})

# Align the input dataframe columns with those expected by the model
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)
    st.write("Prediction (1 = Stroke, 0 = No Stroke):", prediction[0])
