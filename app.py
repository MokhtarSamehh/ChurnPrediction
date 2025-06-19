import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('C:/Users/ROG/Desktop/ITI/DataMining/Day2/churn_dataset.xlsx')

df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

st.title("Customer Churn Prediction App")
age = st.number_input("Enter Age", min_value=0, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=100, value=10)
gender = st.selectbox("Select Gender", options=['Male', 'Female'])

X = df.drop('Churn', axis=1)
y = df['Churn']

model = GaussianNB()
model.fit(X, y)

user_input = pd.DataFrame({
    'Age': [age],
    'Tenure': [tenure],
    'Sex': [1 if gender == 'Male' else 0]
})
prediction = model.predict(user_input)
probability = model.predict_proba(user_input)[0][1]

if prediction[0] == 1:
    result = "The customer is likely to churn."
else:
    result = "The customer is unlikely to churn."

st.write(result)
st.write(f"Probability of churn: {probability:.2f}")