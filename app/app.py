import joblib
import pandas as pd
import streamlit as st
from brief_croise_1.config.conf import CHAMPION_MODEL_PATH, GENDER_OPTIONS, ORDICAL_COLUMNS, YES_NO_OPTIONS


@st.cache_resource
def load_model():
    return joblib.load(CHAMPION_MODEL_PATH)


st.title('Student Mental Health Risk')

if not CHAMPION_MODEL_PATH.exists():
    st.error('Champion model not found, run the mental_health_pipeline DAG first')
    st.stop()

model = load_model()

with st.form('student_form'):
    left, right = st.columns(2)

    with left:
        age = st.number_input('Age', 17, 29, 21)
        gender = st.selectbox('Gender', GENDER_OPTIONS)
        year_of_study = st.number_input('Year of study', 1, 5, 3)
        sleep_hours = st.number_input('Sleep hours', 0.0, 12.0, 6.5)
        study_hours = st.number_input('Study hours per week', 0.0, 40.0, 20.0)
        gpa = st.number_input('GPA', 0.0, 4.0, 2.5)
        social_media_hours = st.number_input('Social media hours', 0.0, 12.0, 6.0)
        screen_time_hours = st.number_input('Screen time hours', 0.0, 16.0, 8.5)

    with right:
        physical_activity = st.selectbox('Physical activity', ORDICAL_COLUMNS['Physical_Activity'])
        diet_quality = st.selectbox('Diet quality', ORDICAL_COLUMNS['Diet_Quality'])
        financial_stress = st.slider('Financial stress', 0.0, 10.0, 6.0)
        family_support = st.slider('Family support', 0.0, 10.0, 6.5)
        attendance_rate = st.slider('Attendance rate (%)', 0.0, 100.0, 85.0)
        part_time_job = st.selectbox('Part time job', YES_NO_OPTIONS)
        previous_history = st.selectbox('Previous mental health history', YES_NO_OPTIONS)

    submitted = st.form_submit_button('Predict')

if submitted:
    student = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Year_of_Study': year_of_study,
        'Sleep_Hours': sleep_hours,
        'Study_Hours_Per_Week': study_hours,
        'GPA': gpa,
        'Social_Media_Hours': social_media_hours,
        'Screen_Time_Hours': screen_time_hours,
        'Physical_Activity': physical_activity,
        'Financial_Stress': financial_stress,
        'Family_Support': family_support,
        'Attendance_Rate': attendance_rate,
        'Part_Time_Job': part_time_job,
        'Diet_Quality': diet_quality,
        'Previous_Mental_Health_History': previous_history,
    }])

    prediction = model.predict(student)[0]
    probabilities = pd.Series(model.predict_proba(student)[0], index=model.classes_)

    st.subheader(f'Predicted risk: {prediction}')
    st.bar_chart(probabilities)
