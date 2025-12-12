# 🧠 ML Engineer Portfolio — Month 1

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)

This repository showcases my **Machine Learning Engineering** skills by building **flexible, reusable pipelines and tools**.  
Focus: Automating common ML tasks such as **data cleaning**, **exploratory data analysis (EDA)**, and **model pipelines**.

---

## 🚀 Completed Projects / Tools

---

### 1️⃣ ML Pipeline Template  
**Location:** `tools/pipeline_template.py`

A reusable **scikit-learn ML workflow template**.

**Features:**

- 🗂 **Load CSV datasets**  
- ⚡ **Preprocess data** (scaling, encoding, handling missing values)  
- 🧪 **Train/test split**  
- 📊 **Model training and evaluation metrics**

**Purpose:**  
Provides a **consistent workflow** for building and testing ML models across datasets, saving time and ensuring best practices.

---

### 2️⃣ Auto Data Cleaner 🧹  
**Location:** `tools/Auto_cleaner.py`

Automatically cleans messy CSV files and generates a **data-quality report**.

**Features:**

- 🔢 Fill missing numeric values with column mean  
- 🏷️ Fill missing categorical values with the most frequent category  
- 🗑️ Remove duplicate rows  
- 📉 Detect and remove outliers using **IQR**  
- 📄 Generate a **JSON report** of cleaning operations  
- 💾 Save cleaned data for downstream tasks

**Usage Example:**

~~~python
from Auto_cleaner import AutoDataCleaner

cleaner = AutoDataCleaner("data/sample_data.csv")
report = cleaner.run_cleaning()
~~~

**Outputs:**

- ✅ `data/cleaned_data.csv`  
- 📄 `reports/cleaning_report.json`

---

### 3️⃣ Automated EDA App 🔍  
**Location:** `auto_eda_app.py`

An **interactive Streamlit app** for automated exploratory data analysis on **CSV or Excel datasets**.

**Features:**

- 📂 Upload and preview datasets  
- 🧮 Generate **statistical summaries** for numeric and categorical columns  
- 📊 Display **categorical value counts**  
- 📈 Visualizations:  
  - 🟦 Histograms  
  - 📦 Boxplots  
  - 🌐 Correlation heatmaps  
- 🎛️ User-selectable analysis options  
- 📥 Download full EDA report as a **ZIP** containing CSV summaries and plots

**Usage:**

~~~bash
streamlit run auto_eda_app.py
~~~

---

## 🗂 Project Structure

```
/projects   # Future mini-projects
/tools      # Scripts, Auto_cleaner, pipeline templates
/data       # Sample datasets
/models     # Saved ML models/templates
/notebooks  # Exploratory analysis or experiments
/reports    # Data cleaning or EDA reports
```

---

## 🎯 Purpose of This Repository

- 💡 Demonstrate **hands-on ML engineering skills**  
- 🧹 Maintain a **clean, professional project structure**  
- 🔧 Build **reusable automation tools**  
- 🚀 Automate repetitive ML tasks:  
  - 🧹 Data cleaning  
  - 🔍 Exploratory data analysis  
  - ⚡ ML pipeline creation

---

## 🔜 Next Steps (Month 2+)

- 🛠️ Add **feature engineering** + **hyperparameter tuning**  
- 📌 Implement **model versioning & tracking**  
- 🌐 Build end-to-end ML projects  
- 🤖 Add advanced ML/AI techniques for real-world datasets  
