# auto_cleaner_app.py
import streamlit as st
import pandas as pd
from Auto_cleaner import AutoDataCleaner

st.set_page_config(page_title="Auto Data Cleaner", layout="wide")
st.title("🧹 Auto Data Cleaner")

# -------------------------------
# 1. Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Original Dataset")
    st.dataframe(df)

    # -------------------------------
    # 2. Initialize cleaner
    # -------------------------------
    cleaner = AutoDataCleaner(df=df)

    # Optional: let user choose cleaning options
    st.subheader("Cleaning Options")
    numeric_strategy = st.selectbox("Missing values (numeric columns)", ["mean", "median"])
    categorical_strategy = st.selectbox("Missing values (categorical columns)", ["most_frequent"])

    detect_outliers_method = st.selectbox("Outlier detection method", ["zscore", "iqr"])
    outlier_threshold = st.number_input("Z-score threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.1)

    # -------------------------------
    # 3. Run cleaning
    # -------------------------------
    if st.button("Clean Data"):
        cleaner.handle_missing(numeric_strategy, categorical_strategy)
        cleaner.remove_duplicates()
        cleaner.detect_outliers(method=detect_outliers_method, threshold=outlier_threshold)
        cleaned_df = cleaner.df
        report = cleaner.generate_report()

        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_df)

        st.subheader("Data Cleaning Report")
        st.json(report)

        # -------------------------------
        # 4. Download cleaned data
        # -------------------------------
        csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
