# 🧠 ML Engineer Portfolio — Month 1

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)

This repository demonstrates my hands-on **Machine Learning Engineering** skills by building **flexible, reusable pipelines and tools**.  
The focus is on automating common ML tasks such as **data cleaning**, **exploratory data analysis (EDA)**, and **model pipelines**.

---

## 🚀 Completed Projects / Tools

### 1️⃣ ML Pipeline Template
**Location:** `tools/pipeline_template.py`

A reusable **scikit-learn workflow template** for ML projects.

**Features:**

- Load CSV datasets
- Preprocess data (scaling, encoding, handle missing values)
- Train/test split
- Model training and evaluation metrics

**Purpose:**  

Provides a **consistent ML workflow** for building and testing models across datasets, saving time and ensuring best practices.

---

### 2️⃣ Auto Data Cleaner 🧹
**Location:** `tools/Auto_cleaner.py`

Automatically cleans messy CSV files and generates a **data-quality report**.

**Features:**

- Fill missing numeric values with column mean
- Fill missing categorical values with the most frequent category
- Remove duplicate rows
- Detect and remove outliers using **IQR**
- Generate a **JSON report** of cleaning operations
- Save cleaned data for downstream tasks

**Usage Example:**

```python
from Auto_cleaner import AutoDataCleaner

cleaner = AutoDataCleaner("data/sample_data.csv")
report = cleaner.run_cleaning()
