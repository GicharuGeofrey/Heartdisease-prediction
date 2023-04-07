import pandas as pd
import numpy as np
import pickle
import streamlit as st


st.write("""
# Heart Disease Prediction App
This App predicts the **PROBABILITY** of a heart disease
         
"""
        )


st.sidebar.header("User Input Header")


def user_input_feature():
        age = st.sidebar.slider('Age (yrs)',1,100,52)
        sex = st.sidebar.slider('sex',0,1,1)
        cp = st.sidebar.slider('Chest Pain',0,3,0)
        trestbps = st.sidebar.slider('Resting Blood Pressure(mmHg)',90,200,125)
        chol = st.sidebar.slider('Cholestral(mg/dl)',100,400,212)
        fbs= st.sidebar.slider('fasting blood sugar',0,1,0)
        restecg = st.sidebar.slider('resting electrocardiographic measurement',0,2,1)
        thalach = st.sidebar.slider('Maximum heartrate',100,371,168)
        exang = st.sidebar.slider('exercise induced angina',0,1,0)
        oldpeak = st.sidebar.slider('Old Peak',0.0,7.0,1.0)
        slope = st.sidebar.slider('Slope',0,2,2)
        ca = st.sidebar.slider('Number of major blood vessels',0,3,2)
        thal = st.sidebar.slider('Thalassemia',0,3,3)
        data = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
        features = pd.DataFrame(data, index=[0])
        return features
inputDf = user_input_feature()

dff = pd.read_csv('/home/linuxdev/coding/heart.csv')
df = dff.drop(['target'], axis =1 )


st.subheader('User Input Features')
st.write(df)

model_logreg = pickle.load(open('model_logreg.pkl','rb'))
iprediction = model_logreg.predict(inputDf)
prediction = model_logreg.predict(df)
prediction_proba = model_logreg.predict_proba(df)
iprediction_proba = model_logreg.predict_proba(inputDf)

st.sidebar.subheader('Prediction')
heart_disease = np.array(['At risk of heart disease','At NO risk of heart disease'])
st.sidebar.write(heart_disease[iprediction])
st.sidebar.header('Prediction Probability')
st.sidebar.write(iprediction_proba)

st.subheader('Prediction')
heart_disease = np.array(['At risk of heart disease','At NO risk of heart disease'])
st.write(heart_disease[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)







