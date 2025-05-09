🛍️ Product Purchase Prediction App (Logistic Regression)
This is a Machine Learning web app built with Streamlit that predicts whether a person is likely to purchase a product based on real-life features such as age, gender, occupation, income, and product category.

🚀 Features
Predicts product purchase behavior using Logistic Regression.

Takes multiple input features:
✅ Age
✅ Gender
✅ Occupation
✅ Income
✅ Product Category

Automatically retrains the model using the dataset every time the app runs (no .pkl file needed).

Shows both predicted label and prediction confidence (probability).

📁 Dataset
The model uses a CSV file named purchase_data.csv that contains 10,000 real-life-style entries with the following columns:

Age (int): 15–65

Gender (str): Male or Female

Occupation (str): Student, Employee, Self-Employed, Retired

Income (int): Annual income (e.g. 10000–100000)

Product_Category (str): Examples: Cosmetics, Electronics, Clothing, Books, Fitness, Food, etc.

Purchased (int): 1 = Purchased, 0 = Not Purchased

⚙️ How to Run
Step 1: Clone the repository or download the files
bash
Copy
Edit
git clone https://github.com/yourusername/purchase-predictor
cd purchase-predictor
Step 2: Install dependencies
bash
Copy
Edit
pip install streamlit pandas scikit-learn
Step 3: Run the app
bash
Copy
Edit
streamlit run app.py
Make sure purchase_data.csv is in the same folder as app.py.

🧪 Model Details
Algorithm: Logistic Regression

Trained live on every app load

Preprocessing: Label encoding for categorical features

📸 Sample Output
makefile
Copy
Edit
Prediction: Will Purchase
Probability: 87.5%
🛠 Tech Stack
Python

Streamlit

scikit-learn

pandas

🙋‍♀️ Example Use Case
This tool can help e-commerce platforms or marketing teams predict customer buying behavior based on demographic data and product types.

