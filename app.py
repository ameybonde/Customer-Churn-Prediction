import streamlit as st
import pandas as pd
import pickle

model=pickle.load(open("model.pkl","rb"))
preprocessor=pickle.load(open("preprocessor.pkl","rb"))

st.title("Customer Churn Prediction")

Gender=st.selectbox("Gender",['Male','Female'])
SeniorCitizen=st.selectbox("Senior Citizen",[0,1])
Partner=st.selectbox("Partner",['Yes','No'])
Dependents=st.selectbox("Dependents",['Yes','No'])
tenure=st.number_input("Tenure",min_value=0,max_value=72,value=12)
PhoneService=st.selectbox("Phone Service",['Yes','No'])
MultipleLines=st.selectbox("Multiple Lines",['Yes','No','No phone service'])
InternetService=st.selectbox("Internet Service",['DSL','Fiber optic','No'])
OnlineSecurity=st.selectbox("Online Security",['Yes','No','No internet service'])
OnlineBackup=st.selectbox("Online Backup",['Yes','No','No internet service'])
DeviceProtection=st.selectbox("Device Protection",['Yes','No','No internet service'])
TechSupport=st.selectbox("Tech Support",['Yes','No','No internet service'])
StreamingTV=st.selectbox("Streaming TV",['Yes','No','No internet service'])
StreamingMovies=st.selectbox("Streaming Movies",['Yes','No','No internet service'])
Contract=st.selectbox("Contract",['Month-to-month','One year','Two year'])
PaperlessBilling=st.selectbox("Paperless Billing",['Yes','No'])
PaymentMethod=st.selectbox("Payment Method",['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
MonthlyCharges=st.number_input("Monthly Charges",min_value=0.0,value=70.0)
TotalCharges=st.number_input("Total Charges",min_value=0.0,value=2000.0)

if st.button("Predict"):
    input_data=pd.DataFrame({
        "gender":[Gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":[OnlineSecurity],
        "OnlineBackup":[OnlineBackup],
        "DeviceProtection":[DeviceProtection],
        "TechSupport":[TechSupport],
        "StreamingTV":[StreamingTV],
        "StreamingMovies":[StreamingMovies],
        "Contract":[Contract],
        "PaperlessBilling":[PaperlessBilling],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges],
        "TotalCharges":[TotalCharges]
    })

    #Preprocess input data
    input_processed=preprocessor.transform(input_data)

    #Prediction
    prediction=model.predict(input_processed)

    #Probability
    probability = model.predict_proba(input_processed)[0][1]

    if prediction==1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")
    
    st.write(f"Churn Probability: {probability:.2f}")