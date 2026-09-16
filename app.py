import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")


@st.cache_resource
def load_bundle():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reference_data():
    df = pd.read_csv("customer_churn_dataset-master.csv")
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
    return df


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
numerical_columns = bundle["numerical_columns"]
categorical_columns = bundle["categorical_columns"]
feature_columns = bundle["feature_columns"]
model_name = bundle["model_name"]

df_ref = load_reference_data()

st.title("📉 Customer Churn Prediction")
st.caption(f"Model in use: **{model_name}**")
st.write("Enter the customer's details below to predict whether they are likely to churn.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        st.subheader("Numerical info")
        for col in numerical_columns:
            default_val = float(df_ref[col].median())
            inputs[col] = st.number_input(col, value=default_val, step=1.0)

    with col2:
        st.subheader("Categorical info")
        for col in categorical_columns:
            options = sorted(df_ref[col].dropna().unique().tolist())
            inputs[col] = st.selectbox(col, options)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])

    input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn.")
    else:
        st.success(f"✅ This customer is likely to stay.")