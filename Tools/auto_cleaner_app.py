import sys
from pathlib import Path
import streamlit as st
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Tools.Auto_cleaner import AutoDataCleaner

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Auto Data Cleaner", layout="wide")
st.title("Auto Data Cleaner")

# -------------------------------------------------------
# 1. Upload Dataset
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)", 
    type=["csv", "xlsx"]
)

if uploaded_file:

    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Original Dataset")
    st.dataframe(df)

    # -------------------------------------------------------
    # 2. Initialize Cleaner
    # -------------------------------------------------------
    cleaner = AutoDataCleaner(df=df)

    # -------------------------------------------------------
    # Cleaning Options
    # -------------------------------------------------------
    st.subheader("Cleaning Options")

    numeric_strategy = st.selectbox(
        "Missing values (numeric columns)", 
        ["mean", "median"]
    )

    categorical_strategy = st.selectbox(
        "Missing values (categorical columns)", 
        ["most_frequent"]
    )

    detect_outliers_method = st.selectbox(
        "Outlier detection method", 
        ["zscore", "iqr"]
    )

    outlier_threshold = st.number_input(
        "Z-score threshold", 
        min_value=1.0, max_value=10.0, 
        value=3.0, 
        step=0.1
    )

    # -------------------------------------------------------
    # 3. Run Cleaning
    # -------------------------------------------------------
    if st.button("Clean Data"):

        # Step 1: Standardize column types cleanly
        cleaner.coerce_types()

        # Step 2: Handle missing values
        cleaner.handle_missing(
            numeric_strategy=numeric_strategy, 
            categorical_strategy=categorical_strategy
        )

        # Step 3: Remove duplicates
        cleaner.remove_duplicates()

        # Step 4: Outlier detection
        cleaner.detect_outliers(
            method=detect_outliers_method, 
            threshold=outlier_threshold
        )

        # Final cleaned data
        cleaned_df = cleaner.df
        report = cleaner.generate_report()

        # -------------------------------------------------------
        # 4. Show Output
        # -------------------------------------------------------
        st.subheader("Cleaned Dataset")
        st.dataframe(cleaned_df)

        st.subheader("Data Cleaning Report")
        st.json(report)

        # -------------------------------------------------------
        # 5. Download File
        # -------------------------------------------------------
        csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
