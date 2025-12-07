import pandas as pd
import numpy as np
from pathlib import Path

class AutoDataCleaner:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.df = pd.read_csv(self.filepath)
        self.report = {}

    def handle_missing(self, numeric_strategy="mean", categorical_strategy="most_frequent"):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        cat_cols = self.df.select_dtypes(exclude=np.number).columns

        for col in numeric_cols:
            if numeric_strategy == "mean":
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif numeric_strategy == "median":
                self.df[col].fillna(self.df[col].median(), inplace=True)

        for col in cat_cols:
            if categorical_strategy == "most_frequent":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        self.report["duplicates_removed"] = before - after
        return self

    def detect_outliers(self, method="zscore", threshold=3):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        outlier_counts = {}
        for col in numeric_cols:
            if method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_counts[col] = (z_scores > threshold).sum()
            elif method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_counts[col] = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
        self.report["outliers_detected"] = outlier_counts
        return self

    def generate_report(self):
        self.report["shape"] = self.df.shape
        self.report["missing_values"] = self.df.isna().sum().to_dict()
        return self.report

    def save_cleaned(self, output_path: str):
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
        return self

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    cleaner = AutoDataCleaner("../Data/Sample_data.csv")
    cleaner.handle_missing().remove_duplicates().detect_outliers()
    report = cleaner.generate_report()
    print("Data Cleaning Report:", report)
    cleaner.save_cleaned("../Data/Sample_data_cleaned.csv")