# auto_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AutoEDA:
    def __init__(self, file_path, output_folder="EDA_Report"):
        self.file_path = file_path
        self.output_folder = output_folder
        self.df = None
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading file: {e}")

    def basic_info(self):
        print("\n--- Dataset Info ---")
        print(self.df.info())
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        print("\n--- Statistical Summary ---")
        print(self.df.describe(include='all').transpose())
        self.df.describe(include='all').transpose().to_csv(os.path.join(self.output_folder, "statistical_summary.csv"))

    def categorical_summary(self):
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        summary = {}
        for col in cat_cols:
            summary[col] = self.df[col].value_counts()
            summary[col].to_csv(os.path.join(self.output_folder, f"{col}_value_counts.csv"))
        print(f"\nCategorical columns summary saved: {list(cat_cols)}")

    def visualize_data(self):
        # Histograms for numeric columns
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.savefig(os.path.join(self.output_folder, f"{col}_histogram.png"))
            plt.close()

        # Boxplots for numeric columns
        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.savefig(os.path.join(self.output_folder, f"{col}_boxplot.png"))
            plt.close()

        # Correlation heatmap
        plt.figure(figsize=(8,6))
        corr = self.df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(self.output_folder, "correlation_heatmap.png"))
        plt.close()

        print(f"\nVisualizations saved in folder: {self.output_folder}")

    def run_eda(self):
        self.load_data()
        self.basic_info()
        self.categorical_summary()
        self.visualize_data()
        print("\nEDA completed successfully.")

# Example usage
# eda = AutoEDA("your_dataset.csv")
# eda.run_eda()
