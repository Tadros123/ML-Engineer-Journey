ML Engineer Portfolio — Month 1

This repository showcases ML projects and reusable tools I’ve developed to build a strong foundation in machine learning engineering.
The focus is on creating flexible pipelines and scripts that can be applied to a variety of datasets, with automated data cleaning and exploratory data analysis (EDA).

Completed Projects / Tools
1. ML Pipeline Template

Location: tools/pipeline_template.py

A reusable machine learning workflow template built with scikit-learn.

Features:

Load CSV datasets

Preprocess data (scaling, encoding, handling missing values)

Train/test split

Model training and evaluation metrics

Purpose:
Provides a consistent structure for building and testing ML models across different datasets, saving time and ensuring best practices.

2. 🧹 Auto Data Cleaner

Location: tools/Auto_cleaner.py

A small tool that cleans messy CSV files automatically.

Features:

Fills missing numeric values using the column mean

Fills missing categorical values using the most frequent category

Removes duplicate rows

Detects and removes outliers using IQR

Generates a data-quality report

Saves both cleaned data and a JSON report

How It Works:

from Auto_cleaner import AutoDataCleaner

cleaner = AutoDataCleaner("data/sample_data.csv")
report = cleaner.run_cleaning()


This automatically creates:

data/cleaned_data.csv

reports/cleaning_report.json

3. 🔍 Automated EDA App

Location: auto_eda_app.py

An interactive Streamlit app that performs automated exploratory data analysis on CSV or Excel datasets.

Features:

Load CSV or Excel files

Display raw data

Generate statistical summaries for numeric and categorical columns

Display categorical value counts interactively

Automatically generate visualizations:

Histograms (numeric columns)

Boxplots (numeric columns)

Correlation heatmaps

Optional selection of which visualizations to generate

Download the full EDA report as a ZIP file containing CSV summaries and plots

How It Works:
Run the Streamlit app:

streamlit run auto_eda_app.py


Upload your dataset, select visualizations, and view results interactively. The complete EDA report can be downloaded as a ZIP for offline use or sharing.

Project Structure
/projects   # Future mini-projects
/tools      # Scripts, Auto_cleaner, pipeline templates
/data       # Sample datasets
/models     # Saved ML models/templates
/notebooks  # Exploratory analysis or experiments
/reports    # Data cleaning or EDA reports

Purpose of This Repository

Demonstrate hands-on ML skills

Maintain clean project structure

Build reusable tools that can be applied to real-world datasets

Showcase the ability to automate repetitive ML tasks (cleaning, EDA, pipeline building)
