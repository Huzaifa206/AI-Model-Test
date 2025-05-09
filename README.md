# 🛍️ Product Purchase Prediction App (Logistic Regression)

A smart and intuitive machine learning web application built with **Streamlit** that predicts whether a person is likely to purchase a product based on real-world attributes such as **age, gender, occupation, income**, and **product category**.

---

## 📊 Overview

This application utilizes **Logistic Regression** to predict purchase behavior based on various inputs. The model is trained in real-time using a CSV file with synthetic yet realistic data of 10,000 people. The app is simple, interactive, and helpful for understanding how customer segments might interact with different product categories.

---

## 🚀 Features

✅ Real-time predictions  
✅ Intuitive Streamlit user interface  
✅ Input fields for Age, Gender, Occupation, Income, and Product Category  
✅ Automatically retrains Logistic Regression model every time  
✅ Outputs prediction along with probability scores  
✅ No external `.pkl` model file required

---

## 📁 Dataset Structure (`purchase_data.csv`)

| Column Name        | Description                                 |
|--------------------|---------------------------------------------|
| `Age`              | Age of the person (e.g., 18–65)             |
| `Gender`           | Male or Female                              |
| `Occupation`       | Student, Employee, Self-Employed, Retired   |
| `Income`           | Annual income in USD (e.g., 10000–100000)   |
| `Product_Category` | Type of product (e.g., Cosmetics, Clothing) |
| `Purchased`        | 1 = Purchased, 0 = Not Purchased             |

---
🧠 Model Details
Algorithm: Logistic Regression
Training: Performed live on app launch using the dataset
Preprocessing:
Label encoding for categorical variables (Gender, Occupation, Product_Category)
Feature scaling using StandardScaler

📦 Tech Stack
Python 🐍
Streamlit 🌐
scikit-learn 🤖
pandas 🐼

🎯 Use Case
This project can be adapted by:
E-commerce companies to predict customer buying behavior
Marketers for audience segmentation and targeting
Beginners learning logistic regression in a practical way

