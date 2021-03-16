import streamlit as st
import pandas as pd
import numpy as np
import pickle

def value(lst,string):
    for i in range(len(lst)):
        if lst[i]==string:
            return i
sex=['Female','Male']
edu=['10th pass','12th pass/Diploma','Bachelors','Masters or Higher']
yn=['NO','YES']

st.write("""
# Lifely
**Heart Disease Diagonistic App**

Try it for free.

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Input your data here 
""")
def user_input_features():
        male = st.sidebar.selectbox('Sex',('Female','Male'))
        age= st.sidebar.slider('Age',5.0,100.0,30.0)
        education = st.sidebar.selectbox('Education',('10th pass','12th pass/Diploma','Bachelors','Masters or Higher'))
        current_smoker = st.sidebar.selectbox('Current Smoker',('NO','YES'))
        cigsPerDay = st.sidebar.slider('Cigarettes per Day',0,100,20)
        BPMeds = st.sidebar.selectbox('Takes BP medicines',('NO','YES'))
        prevstrk = st.sidebar.selectbox('Had any prevalent Stroke',('NO','YES'))
        prevhyp = st.sidebar.selectbox('Had any prevalent Hypertension',('NO','YES'))
        diabetes = st.sidebar.selectbox('Have diabetes',('NO','YES'))
        chol = st.sidebar.slider('Cholesterol (mg/dl)',0.0,700.0,230.0)
        highbp = st.sidebar.slider('Blood Pressure(upper value) (mmHg)',100.0,250.0,120.0)
        lowbp = st.sidebar.slider('Blood Pressure(Lower Value) (mmHg)',50.0,180.0,80.0)
        BMI = st.sidebar.slider('BMI (kg/m^2)',15.0,70.0,23.0)
        heart_rate = st.sidebar.slider('Heart Rate (per minute)',30.0,130.0,40.0)
        glucose = st.sidebar.slider('Glucose (mg/dl)',100.0 ,500.0,110.0)
        data = {'male':value(sex,male),
                'age':age,
                'education':value(edu,education),
                'currentSmoker':value(yn,current_smoker),
                'cigsPerDay':cigsPerDay,
                'BPMeds':value(yn,BPMeds),
                'prevalentStroke':value(yn,prevstrk),
                'prevalentHyp':value(yn,prevhyp),
                'diabetes':value(yn,diabetes),
                'totChol':chol,
                'sysBP':highbp,
                'diaBP':lowbp,
                'BMI':BMI,
                'heartRate':heart_rate,
                'glucose':glucose}
        features = pd.DataFrame(data, index=[0])
        return features
df=user_input_features()
st.write(df.T)
    

# Reads in saved classification model
load_clf = pickle.load(open('heart_disease.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
if prediction==0:
    st.write("""## You don't have any heart problem ☺️""")
else:
    st.write("""## Go to a doctor.You have heart problems.""")

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
         
         Developed with ❤️ by *SAGNIK ROY*
         
         Follow me on :[github](https://github.com/sagnik1511)
         
         Visit my other ML/DL works in [Kaggle](https://kaggle.com/sagnik1511/notebooks)
         
         For any queries email me on ***sagnik.jal00@gmail.com***
         
         All rights reserved.
         """)