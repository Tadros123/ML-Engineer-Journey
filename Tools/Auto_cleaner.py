import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats
from typing import Optional, List, Dict, Any


class AutoDataCleaner:
    def __init__(self, filepath: str = None, df: pd.DataFrame = None):
        """
        Initialize AutoDataCleaner with either a file path or a DataFrame.
        """
        if df is not None:
            self.df = df.copy()
        elif filepath is not None:
            script_dir = Path(__file__).parent
            self.filepath = (script_dir / filepath).resolve()
            if not self.filepath.exists():
                raise FileNotFoundError(f"{self.filepath} does not exist")
            self.df = pd.read_csv(self.filepath)
        else:
            raise ValueError("You must provide either a filepath or a DataFrame")

        self.original_df = self.df.copy()
        self.report = {
            "initial_shape": self.df.shape,
            "duplicates_removed": 0,
            "missing_filled": {},
            "outliers_detected": {},
            "invalid_emails": [],
            "type_conversions": {},
        }

    # -----------------------------------------------------------
    # 1. Utility helpers
    # -----------------------------------------------------------

    def _strip_currency(self, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        cleaned = cleaned.replace("", np.nan)
        return cleaned

    def _text_to_number(self, s: str) -> Optional[float]:
        if pd.isna(s):
            return None
        s = str(s).lower().strip()

        mapping = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
        }

        if s in mapping:
            return float(mapping[s])

        parts = s.split()
        total = 0
        found = False
        for p in parts:
            if p in mapping:
                total += mapping[p]
                found = True

        return float(total) if found else None

    def _is_email(self, val: str) -> bool:
        if pd.isna(val):
            return False
        pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        return re.match(pattern, str(val).strip()) is not None

    # -----------------------------------------------------------
    # 2. Type coercion & string normalization
    # -----------------------------------------------------------

    def coerce_types(self):
        """
        Automatically convert numeric-looking columns into numeric dtype.
        """
        df = self.df

        for col in df.columns:
            original_dtype = str(df[col].dtype)

            # Try currency stripping
            cleaned = self._strip_currency(df[col])
            coerced = pd.to_numeric(cleaned, errors="coerce")

            # Try word-based numbers if still NaN
            mask = coerced.isna() & cleaned.notna()
            if mask.any():
                mapped = cleaned[mask].map(self._text_to_number)
                coerced.loc[mask] = mapped

            # Apply numeric conversion only if it converts significant portion of the data
            before_nonnull = df[col].dropna().shape[0]
            after_numeric = coerced.dropna().shape[0]

            if before_nonnull == 0:
                continue

            if after_numeric / before_nonnull >= 0.40:
                df[col] = coerced
                self.report["type_conversions"][col] = {"from": original_dtype, "to": "numeric"}
            else:
                # Clean stray "nan" strings and whitespace
                df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

        self.df = df

    # -----------------------------------------------------------
    # 3. Duplicate removal
    # -----------------------------------------------------------

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        self.report["duplicates_removed"] = removed
        return self

    # -----------------------------------------------------------
    # 4. Missing value handling
    # -----------------------------------------------------------

    def handle_missing(self, numeric_strategy="mean", categorical_strategy="most_frequent"):
        df = self.df

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns

        filled = {}

        # Numeric columns
        for col in numeric_cols:
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                if numeric_strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = df[col].median()

                df[col].fillna(fill_val, inplace=True)
                filled[col] = int(missing_before)

        # Categorical columns
        for col in categorical_cols:
            missing_before = df[col].isna().sum()
            if missing_before > 0 and categorical_strategy == "most_frequent":
                try:
                    mode_val = df[col].mode(dropna=True).iloc[0]
                    df[col].fillna(mode_val, inplace=True)
                    filled[col] = int(missing_before)
                except Exception:
                    pass

        self.report["missing_filled"] = filled
        return self

    # -----------------------------------------------------------
    # 5. Outlier detection
    # -----------------------------------------------------------

    def detect_outliers(self, method="zscore", threshold=3):
        df = self.df
        numeric_cols = df.select_dtypes(include=np.number).columns

        outliers = {}

        for col in numeric_cols:
            col_data = df[col].dropna()
            if col_data.shape[0] < 2:
                outliers[col] = 0
                continue

            if method == "zscore":
                z = np.abs(stats.zscore(col_data))
                outliers[col] = int((z > threshold).sum())

            elif method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                mask = (col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)
                outliers[col] = int(mask.sum())

        self.report["outliers_detected"] = outliers
        return self

    # -----------------------------------------------------------
    # 6. Email validation
    # -----------------------------------------------------------

    def find_invalid_emails(self):
        invalids = []
        for col in self.df.columns:
            if "email" in col.lower():
                col_data = self.df[col]
                invalids = [v for v in col_data.dropna().unique() if not self._is_email(v)]
                break

        self.report["invalid_emails"] = invalids
        return invalids

    # -----------------------------------------------------------
    # 7. Report
    # -----------------------------------------------------------

    def generate_report(self):
        self.report["final_shape"] = self.df.shape
        self.report["missing_after"] = self.df.isna().sum().to_dict()
        return self.report

    # -----------------------------------------------------------
    # 8. Simple clean() API for backward compatibility
    # -----------------------------------------------------------

    def clean(self):
        """
        Basic cleaning (duplicates only).
        Provided for backward compatibility.
        """
        self.remove_duplicates()
        return self.df

    def save_cleaned(self, output_path: str):
        self.df.to_csv(output_path, index=False)
        return self

