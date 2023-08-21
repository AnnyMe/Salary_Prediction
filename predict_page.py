import streamlit as st
import pickle
import numpy as np
import joblib


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
print("Successfully loaded model")


regressor_loaded = data["model"]
LabelEncoder_Country = data["LabelEncoder_Country"]
LabelEncoder_DevType = data["LabelEncoder_DevType"]


def show_predict_page():
    st.title("IT Role Salary Prediction")

    st.write("""### Necessary information are needed to predict salary""")

    countries = (
        'United States of America',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'India',
        'Canada',
        'France',
        'Brazil',
        'Spain',
        'Netherlands',
        'Australia')

    DevTypes = (
        'Developer, full-stack',
        'Developer, back-end',
        'Developer, front-end',
        'Developer, desktop or enterprise applications',
        'Developer, mobile',
        'Other',
        'Engineering manager',
        'Developer, embedded applications or devices',
        'Data scientist or machine learning specialist',
        'Developer, front-end;Developer, full-stack;Developer, back-end',
        'Engineer, data',
        'DevOps specialist',
        'Developer, full-stack;Developer, back-end',
        'Senior Executive (C-Suite, VP, etc.)',
        'Academic researcher',
        'Research & Development role',
        'Cloud infrastructure engineer',
        'Data or business analyst',
        'Developer, front-end;Developer, full-stack',
        'Developer, QA or test',
        'System administrator',
        'Developer, game or graphics',
        'Developer, back-end;Developer, desktop or enterprise applications',
        'Project manager',
        'Engineer, site reliability',
        'Product manager',
        'Security professional',
        'Developer, front-end;Developer, full-stack;Developer, back-end;Developer, desktop or enterprise applications',
        'Scientist',
        'Developer, back-end;Cloud infrastructure engineer'
    )

    educations = (
        "Primary/elementary school",
        "Associate degree (A.A., A.S., etc.)",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
        "Some college/university study without earning a degree",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
        "Professional degree (JD, MD, etc.)",
        "Other doctoral degree (Ph.D., Ed.D., etc.)",
        "Something else"
    )

    country = st.selectbox("Country", countries)
    DevType = st.selectbox(
        "Choose one which best describe the job role", DevTypes)
    education = st.selectbox("Education level", educations)
    work_experience = st.slider(
        "Years of work experience", min_value=0, max_value=50, value=5)
    code_experience = st.slider(
        "Years of code experience", min_value=0, max_value=50, value=5)
    start_calculate = st.button("Calculate Salary")
    if start_calculate == True:
        dic_EdLevel = {"Something else": 1, "Primary/elementary school": 1, "Associate degree (A.A., A.S., etc.)": 1, "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1, "Some college/university study without earning a degree": 1,
                       "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 2, "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 3, "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 4, "Professional degree (JD, MD, etc.)": 4, "Other doctoral degree (Ph.D., Ed.D., etc.)": 4}
        X = np.array(
            [[education, code_experience, DevType, country, work_experience]])
        X[:, 3] = LabelEncoder_Country.transform(X[:, 3])
        X[:, 2] = LabelEncoder_DevType.transform(X[:, 2])
        X[0, 0] = dic_EdLevel[X[0, 0]]

        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
