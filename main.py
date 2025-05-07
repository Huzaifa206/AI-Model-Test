import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("Product Purchase Prediction App (Logistic Regression)")

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 23, 31, 35],
    'Salary': [19000, 21000, 50000, 60000, 57000, 78000, 80000, 22000, 28000, 35000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 0, 0, 1]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Features and Target variable
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display model performance
st.subheader("Model Evaluation")
st.write(f'**Accuracy:** {accuracy * 100:.2f}%')
st.write("**Confusion Matrix:**")
st.write(conf_matrix)

# User input for prediction
st.subheader("Predict Purchase Based on Age and Salary")

try:
    age = st.number_input("Enter Age:", min_value=0, max_value=100, value=25)
    salary = st.number_input("Enter Salary:", min_value=0, max_value=200000, value=30000)

    # When user clicks the Predict button
    if st.button("Predict"):
        user_input = scaler.transform([[age, salary]])
        prediction = model.predict(user_input)
        probabilities = model.predict_proba(user_input)

        if prediction[0] == 1:
            st.success("The person is predicted to **purchase** the product.")
        else:
            st.info("The person is predicted **not to purchase** the product.")

        st.write(f"**Probability of Not Purchasing (0):** {probabilities[0][0] * 100:.2f}%")
        st.write(f"**Probability of Purchasing (1):** {probabilities[0][1] * 100:.2f}%")

except ValueError:
    st.error("Please enter valid numbers for Age and Salary.")
