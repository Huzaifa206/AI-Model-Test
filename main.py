import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample expanded dataset
data = {
    'Age': [58, 25, 19, 65, 35, 45, 30, 27, 50, 60],
    'Salary': [124974, 72787, 132757, 87926, 16703, 54000, 33000, 29500, 90000, 100000],
    'Gender': ['Female', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male'],
    'MaritalStatus': ['Married', 'Single', 'Married', 'Married', 'Married', 'Single', 'Single', 'Single', 'Married', 'Single'],
    'JobType': ['Business', 'Student', 'Business', 'Retired', 'Employee', 'Employee', 'Student', 'Employee', 'Business', 'Retired'],
    'Purchased': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Age', 'Salary', 'Gender', 'MaritalStatus', 'JobType']]
y = df['Purchased']

# Categorical columns
categorical_cols = ['Gender', 'MaritalStatus', 'JobType']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Random Forest pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("🛍️Product Purchase Prediction App (Logistic Regression)")

st.markdown("This machine learning model predicts whether a person is likely to purchase a product based on multiple factors such as age, salary, job type, education level, and other personal details. It uses a Logistic Regression algorithm trained on structured data to classify users and estimate the probability of purchase. This tool demonstrates how predictive models can assist in customer profiling and targeted marketing strategies.")

# Input form
age = st.number_input("Enter Age:", min_value=18, max_value=100, value=30)
salary = st.number_input("Enter Salary:", min_value=0, max_value=1000000, value=50000)

gender = st.selectbox("Select Gender:", ['Male', 'Female'])
marital_status = st.selectbox("Select Marital Status:", ['Single', 'Married'])
job_type = st.selectbox("Select Job Type:", ['Student', 'Employee', 'Business', 'Retired'])

if st.button("Predict"):
    try:
        # Create DataFrame for prediction
        user_df = pd.DataFrame([{
            'Age': age,
            'Salary': salary,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'JobType': job_type
        }])

        # Predict
        prediction = model.predict(user_df)[0]
        probabilities = model.predict_proba(user_df)[0]

        # Show result
        if prediction == 1:
            st.success("✅ The person is predicted to **purchase** the product.")
        else:
            st.error("❌ The person is predicted **not** to purchase the product.")

        st.markdown(f"**Probability of Not Purchasing (0):** {probabilities[0] * 100:.2f}%")
        st.markdown(f"**Probability of Purchasing (1):** {probabilities[1] * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")
