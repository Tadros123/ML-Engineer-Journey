🧠 ML Engineer Portfolio — Month 1






This repository demonstrates my hands-on Machine Learning Engineering skills by building flexible, reusable pipelines and tools.
The focus is on automating common ML tasks such as data cleaning, exploratory data analysis (EDA), and model pipelines.

🚀 Completed Projects / Tools
1️⃣ ML Pipeline Template

Location: tools/pipeline_template.py

A reusable scikit-learn workflow template for ML projects.

Features:

Load CSV datasets

Preprocess data (scaling, encoding, handle missing values)

Train/test split

Model training and evaluation metrics

Purpose:
Provides a consistent ML workflow for building and testing models across datasets, saving time and ensuring best practices.

2️⃣ Auto Data Cleaner 🧹

Location: tools/Auto_cleaner.py

Automatically cleans messy CSV files and generates a data-quality report.

Features:

Fill missing numeric values with column mean

Fill missing categorical values with the most frequent category

Remove duplicate rows

Detect and remove outliers using IQR

Generate a JSON report of cleaning operations

Save cleaned data for downstream tasks

Usage Example:

from Auto_cleaner import AutoDataCleaner

cleaner = AutoDataCleaner("data/sample_data.csv")
report = cleaner.run_cleaning()


Generates:

data/cleaned_data.csv

reports/cleaning_report.json

3️⃣ Automated EDA App 🔍

Location: auto_eda_app.py

An interactive Streamlit app for automated exploratory data analysis on CSV or Excel datasets.

Features:

Upload and preview datasets

Generate statistical summaries for numeric and categorical columns

Display categorical value counts interactively

Generate visualizations:

Histograms (numeric columns)

Boxplots (numeric columns)

Correlation heatmaps

Optional selection of visualizations to generate

Download the full EDA report as a ZIP containing CSV summaries and plots

Usage:

streamlit run auto_eda_app.py

🗂 Project Structure
/projects   # Future mini-projects
/tools      # Scripts, Auto_cleaner, pipeline templates
/data       # Sample datasets
/models     # Saved ML models/templates
/notebooks  # Exploratory analysis or experiments
/reports    # Data cleaning or EDA reports

🎯 Purpose of This Repository

Demonstrate hands-on ML skills

Maintain a clean, professional project structure

Build reusable automation tools for real-world datasets

Showcase ability to automate repetitive ML tasks:

Data cleaning

Exploratory data analysis

ML pipeline building

🔜 Next Steps (Month 2+)

Extend ML pipeline with feature engineering and hyperparameter tuning

Add model versioning and tracking

Build end-to-end projects combining cleaning, EDA, modeling, and deployment

Apply advanced ML/AI techniques to practical datasets
