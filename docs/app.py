import pickle
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os

# Load the models
with open('decision_tree_model.pickle', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('logistic_regression_model.pickle', 'rb') as file:
    logistic_regression_model = pickle.load(file)

with open('naive_bayes_model.pickle', 'rb') as file:
    naive_bayes_model = pickle.load(file)

# Apply custom CSS
def apply_custom_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def save_prediction(record):
    file_name = 'prediction_records.csv'
    if os.path.exists(file_name):
        records_df = pd.read_csv(file_name)
        records_df = pd.concat([records_df, pd.DataFrame([record])], ignore_index=True)

    else:
        records_df = pd.DataFrame([record])
    records_df.to_csv(file_name, index=False)

def home_page():
    st.title("\U0001F3E0 Credit Card Transaction Fraud Detection")
    st.image("contoh.jpg", use_container_width=True)
    st.write("Welcome to the Credit Card Transaction Fraud Payment Detection app.")

def prediction_page():
    st.title("\U0001F50D Credit Card Transaction Fraud Payment Detection")
    st.subheader('Predict whether a transaction is fraudulent')

    model_choice = st.selectbox("Choose the model for prediction", ["Decision Tree", "Logistic Regression", "Naive Bayes"])
    
    type = st.selectbox("Transaction Type", ["Cash Out", "Payment", "Cash In", "Transfer", "Debit"], key='type')
    amounts = st.number_input("Transaction Amount", min_value=0, max_value=10000000, value=0, step=1, key='amount')
    oldb_orig = st.number_input("Your Balance Before Transaction", min_value=0, max_value=10000000, value=0, step=1, key='old_balance_orig')
    oldb_dest = st.number_input("Recipient's Balance Before Transaction", min_value=0, max_value=10000000, value=0, step=1, key='old_balance_dest')
    newb_orig = st.number_input("Your Balance After Transaction", min_value=0, max_value=10000000, value=0, step=1, key='new_balance_orig')
    newb_dest = st.number_input("Recipient's Balance After Transaction", min_value=0, max_value=10000000, value=0, step=1, key='new_balance_dest')

    type_dict = {"Cash Out": 0, "Payment": 1, "Cash In": 2, "Transfer": 3, "Debit": 4}
    type_numeric = type_dict[type]

    input_data = pd.DataFrame({
        'type': [type_numeric],
        'amount': [amounts],
        'oldbalanceOrg': [oldb_orig],
        'newbalanceOrig': [newb_orig],
        'oldbalanceDest': [oldb_dest],
        'newbalanceDest': [newb_dest]
    })

    if st.button('Predict'):
        try:
            model = decision_tree_model if model_choice == "Decision Tree" else (
                logistic_regression_model if model_choice == "Logistic Regression" else naive_bayes_model)
            
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else None
            
            st.write("### Prediction Results")
            st.write(f"**Prediction:** {'🟢 Not Fraud' if prediction[0] == 0 else '🔴 Fraud'}")
            if prediction_proba is not None:
                st.write(f"**Prediction Probability:** {prediction_proba[0]:.2f}")
            else:
                st.write("**Prediction Probability:** N/A")
            
            st.progress(prediction_proba[0] if prediction_proba is not None else 0.5)
            
            fraud_image_url = "pngtree-fraud-alert-red-rubber-stamp-on-white-insecure-safety-id-vector-png-image_21878902.png"
            not_fraud_image_url = "images.jpeg"
            st.image(fraud_image_url if prediction[0] == 1 else not_fraud_image_url, width=100)
            
            record = {
                "Model": model_choice,
                "Transaction Type": type,
                "Amount": amounts,
                "Old Balance Orig": oldb_orig,
                "New Balance Orig": newb_orig,
                "Old Balance Dest": oldb_dest,
                "New Balance Dest": newb_dest,
                "Prediction": "Fraud" if prediction[0] == 1 else "Not Fraud",
                "Prediction Probability": prediction_proba[0] if prediction_proba is not None else "N/A"
            }
            save_prediction(record)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def records_page():
    st.title("\U0001F4DD Prediction Records")
    file_name = 'prediction_records.csv'
    if os.path.exists(file_name):
        records_df = pd.read_csv(file_name)
        st.write(records_df)
    else:
        st.write("No records found.")

def eda_page():
    st.title("\U0001F4CA Exploratory Data Analysis")
    data = pd.read_csv('Sample-Dataset.csv')
    st.write("### Dataset Overview")
    st.write(data.head())
    
    fig, ax = plt.subplots()
    sns.countplot(data['type'], ax=ax)
    st.pyplot(fig)

    non_numeric_columns = ['nameOrig', 'nameDest']
    data_numeric = data.drop(columns=non_numeric_columns)
    data_encoded = pd.get_dummies(data_numeric, columns=['type'])

    

    st.write("### Distribution of Transaction Amounts")
    fig, ax = plt.subplots()
    sns.histplot(data['amount'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    

def about_page():
    st.title("About - Credit Card Fraud Detection")
    st.write("""
    This application helps to predict whether a financial transaction is fraudulent or not. It uses three different machine learning models:
    - **Decision Tree**: Known for its interpretability and ability to handle non-linear relationships.
    - **Logistic Regression**: A simple and effective baseline model providing probability estimates.
    - **Naive Bayes**: Computationally efficient and handles high-dimensional data well.
    """)



st.sidebar.title("Navigation")
apply_custom_css()
page = st.sidebar.radio("Go to", ["Home", "Prediction", "EDA", "Records", "About"])

if page == "Home":
    home_page()
elif page == "Prediction":
    prediction_page()
elif page == "EDA":
    eda_page()
elif page == "Records":
    records_page()
elif page == "About":
    about_page()

st.markdown("""
<div class="watermark">
    Developed with ❤️ 
</div>
""", unsafe_allow_html=True)