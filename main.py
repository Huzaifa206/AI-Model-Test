import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('purchase_data.csv')

# Preprocess data
le_gender = LabelEncoder()
le_occupation = LabelEncoder()
le_category = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
df['Product_Category'] = le_category.fit_transform(df['Product_Category'])

X = df[['Age', 'Gender', 'Occupation', 'Income', 'Product_Category']]
y = df['Purchased']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🛍️Product Purchase Prediction App (Logistic Regression)")

st.markdown("This machine learning model predicts whether a person is likely to purchase a product based on multiple factors such as age, salary, job type, education level, and other personal details. It uses a Logistic Regression algorithm trained on structured data to classify users and estimate the probability of purchase. This tool demonstrates how predictive models can assist in customer profiling and targeted marketing strategies.")

age = st.slider("Enter Age", 15, 65, 25)
gender = st.selectbox("Select Gender", le_gender.classes_)
occupation = st.selectbox("Select Occupation", le_occupation.classes_)
income = st.number_input("Enter Income", min_value=10000, max_value=100000, step=1000)
category = st.selectbox("Select Product Category", le_category.classes_)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age,
        le_gender.transform([gender])[0],
        le_occupation.transform([occupation])[0],
        income,
        le_category.transform([category])[0]
    ]], columns=['Age', 'Gender', 'Occupation', 'Income', 'Product_Category'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]
    
    st.write(f"### Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")
    st.write(f"### Probability: {round(probability * 100, 2)}%")
