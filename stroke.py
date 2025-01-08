import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Input form (as before)

# Gender
gender = st.selectbox('Select Gender:', ('Select', 'Male', 'Female'))
gender_male = 1 if gender == 'Male' else 0
gender_female = 1 if gender == 'Female' else 0

# Age
age = st.text_input('Enter Age:')
age = pd.to_numeric(age, errors='coerce') if age else None

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
glucose_level = st.text_input('Average Glucose Level (use the guide below if you don\'t know):')
st.write('''
* Normal: 99mg/dl or lower
* Prediabetes: 100mg/dl 125 mg/dl
* Diabetes: 126 mg/dl or higher
''')
glucose_level = pd.to_numeric(glucose_level, errors='coerce') if glucose_level else None

# BMI
bmi = st.text_input('Enter BMI (Body Mass Index):')
st.write('To find your BMI click here: https://www.cdc.gov/bmi/adult-calculator/index.html')
bmi = pd.to_numeric(bmi, errors='coerce') if bmi else None

# Smoking Status
smoker = st.selectbox('Smoking Status/History:', ('Select', 'Use to Smoke', 'Never', 'Smokes', 'Unknown'))
smoking_status_unknown = 1 if smoker == 'Unknown' else 0
smoking_status_formerly_smoked = 1 if smoker == 'Use to Smoke' else 0
smoking_status_never_smoked = 1 if smoker == 'Never' else 0
smoking_status_smokes = 1 if smoker == 'Smokes' else 0

# Load the model and scaler
@st.cache_resource
def load_scaler_and_model():
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
    with open('stroke_xg_boost.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_scaler, loaded_model

scaler, model = load_scaler_and_model()

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

# Load the training dataset
train_data = pd.read_csv('healthcare-dataset-stroke-data.csv')  # Adjust the path to your training dataset

# Filter the training dataset based on Stroke (1 = Stroke, 0 = No Stroke)
stroke_yes_data = train_data[train_data['stroke'] == 1]
stroke_no_data = train_data[train_data['stroke'] == 0]

# Plotting comparison between user data and Stroke Yes/No
def plot_comparison_with_stroke_class(feature_name, user_value, stroke_yes_data, stroke_no_data):
    plt.figure(figsize=(10, 6))

    # Capitalize and remove underscores from the feature name
    feature_name_capitalized = feature_name.replace('_', ' ').title()

     # First plot Stroke No data (red) behind the Stroke Yes data
    sns.histplot(stroke_no_data[feature_name], kde=True, color='red', label=f'Stroke No', alpha=1)

    # Then plot Stroke Yes data (green) to ensure it's in front
    sns.histplot(stroke_yes_data[feature_name], kde=True, color='blue', label=f'Stroke Yes', alpha=1)

    # Plot the user's value (blue dashed line) on top of both
    plt.axvline(user_value, color='blue', linestyle='--', label=f"User's {feature_name_capitalized} ({user_value})")

    # Adding labels and title
    plt.title(f"Comparison of User's {feature_name_capitalized} with Stroke Yes and Stroke No")
    plt.xlabel(feature_name_capitalized)  # Capitalized feature name for the x-axis
    plt.ylabel('Frequency')

    # Capitalizing and updating the legend labels
    plt.legend(title="Stroke Status", loc="best")

    # Show the plot in Streamlit
    st.pyplot(plt)

# Store the comparison plots in session state to prevent them from disappearing
if "plots_displayed" not in st.session_state:
    st.session_state.plots_displayed = False

# Show comparison plots when button is pressed
if st.button("Compare Your Data with Stroke Yes/No"):
    st.session_state.plots_displayed = True
    if age is not None: 
        plot_comparison_with_stroke_class('age', age, stroke_yes_data, stroke_no_data)
    if glucose_level is not None: 
        plot_comparison_with_stroke_class('avg_glucose_level', glucose_level, stroke_yes_data, stroke_no_data)
    if bmi is not None: 
        plot_comparison_with_stroke_class('bmi', bmi, stroke_yes_data, stroke_no_data)

# Display the comparison plots if they were previously displayed
if st.session_state.plots_displayed:
    if age is not None:
        plot_comparison_with_stroke_class('age', age, stroke_yes_data, stroke_no_data)
    if glucose_level is not None:
        plot_comparison_with_stroke_class('avg_glucose_level', glucose_level, stroke_yes_data, stroke_no_data)
    if bmi is not None:
        plot_comparison_with_stroke_class('bmi', bmi, stroke_yes_data, stroke_no_data)

# Store the prediction result to prevent it from disappearing
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Prediction button and displaying the result
if st.button("Predict Stroke Risk"):
    prob = model.predict_proba(input_df)  # Returns an array with probabilities for both classes

    # Probability of stroke (class 1)
    stroke_probability = prob[0][1]  # Second element corresponds to Stroke Yes probability

    # Display the probability
    st.write(f"Probability of having a stroke: {stroke_probability * 100:.2f}%")

    # Display the predicted class (0 or 1)
    prediction = model.predict(input_df)
    st.write("Prediction (1 = Stroke, 0 = No Stroke):", prediction[0])

    # Store the prediction result to ensure button stays visible
    st.session_state.prediction_made = True