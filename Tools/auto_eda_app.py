import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Auto_eda import AutoEDA
import zipfile
from io import BytesIO

st.title("Automated EDA App")

# File upload (CSV or Excel)
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.success(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading file: {e}")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    # Select visualizations to generate
    st.subheader("Select Visualizations to Generate")
    show_hist = st.checkbox("Histograms", value=True)
    show_box = st.checkbox("Boxplots", value=True)
    show_corr = st.checkbox("Correlation Heatmap", value=True)

    if st.button("Run EDA"):
        st.info("Running EDA...")

        # Save uploaded file temporarily
        temp_file = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
        if uploaded_file.name.endswith(".csv"):
            df.to_csv(temp_file, index=False)
        else:
            df.to_excel(temp_file, index=False)
        
        # Run AutoEDA
        eda = AutoEDA(temp_file, output_folder="Streamlit_EDA_Report")
        eda.run_eda()
        st.success("EDA Completed!")

        # Statistical summary
        st.subheader("Statistical Summary")
        summary = pd.read_csv("Streamlit_EDA_Report/statistical_summary.csv", index_col=0)
        st.dataframe(summary)

        # Categorical summaries
        st.subheader("Categorical Summaries")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                st.markdown(f"**{col} Value Counts**")
                value_counts = pd.read_csv(f"Streamlit_EDA_Report/{col}_value_counts.csv", index_col=0)
                st.dataframe(value_counts)
        else:
            st.write("No categorical columns found.")

        # Visualizations based on user selection
        st.subheader("Visualizations")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        if show_hist:
            st.markdown("### Histograms")
            for col in num_cols:
                st.image(f"Streamlit_EDA_Report/{col}_histogram.png", caption=f"Histogram of {col}")

        if show_box:
            st.markdown("### Boxplots")
            for col in num_cols:
                st.image(f"Streamlit_EDA_Report/{col}_boxplot.png", caption=f"Boxplot of {col}")

        if show_corr:
            st.markdown("### Correlation Heatmap")
            st.image("Streamlit_EDA_Report/correlation_heatmap.png", caption="Correlation Heatmap")

        # Create ZIP for download
        st.subheader("Download Full EDA Report")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for root, _, files in os.walk("Streamlit_EDA_Report"):
                for file in files:
                    zip_file.write(os.path.join(root, file), arcname=file)
        st.download_button(
            label="Download EDA Report ZIP",
            data=zip_buffer.getvalue(),
            file_name="EDA_Report.zip",
            mime="application/zip"
        )
