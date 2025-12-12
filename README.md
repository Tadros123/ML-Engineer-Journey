# ML Engineer Portfolio — Month 1

This repository showcases ML projects and reusable tools I’ve developed to build a strong foundation in machine learning engineering.  
The current focus is on creating flexible pipelines and scripts that can be applied to a variety of datasets.

---

## Completed Projects / Tools

### ML Pipeline Template
**Location:** `tools/pipeline_template.py`  

A reusable machine learning workflow template built with `scikit-learn`.  

**Features:**
- Load CSV datasets  
- Preprocess data (scaling, encoding, handle missing values)  
- Train/test split  
- Model training and evaluation metrics  

**Purpose:** Provides a consistent structure for building and testing ML models across different datasets, saving time and ensuring best practices.

### 🧹 Auto Data Cleaner
**Location:** `tools/Auto_cleaner.py`

A small tool that cleans messy CSV files automatically.

**Features**
- Fills missing numeric values using the column mean
- Fills missing categorical values using the most frequent category
- Removes duplicate rows
- Detects and removes outliers using IQR
- Generates a small data-quality report
- Saves both cleaned data and a JSON report

---

🔍 How It Works

You pass it a CSV file:

from Auto_cleaner import AutoDataCleaner

cleaner = AutoDataCleaner("data/sample_data.csv")
report = cleaner.run_cleaning()


This automatically creates:

data/cleaned_data.csv

reports/cleaning_report.json

## Project Structure

/projects # Future mini-projects
/tools # Scripts and pipeline templates
/data # Sample datasets
/models # Saved ML models/templates
/notebooks # Exploratory analysis or experiments


---

This repository is intended to demonstrate **hands-on ML skills**, **clean project structure**, and **reusable tools** that can be applied to real-world projects.

