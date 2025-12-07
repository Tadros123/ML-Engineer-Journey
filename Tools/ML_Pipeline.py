import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
@dataclass
class PipelineConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    log_level: str = "INFO"

    def __post_init__(self):
        self.model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": self.random_state
        }


# --------------------------------------------------------
# LOGGER
# --------------------------------------------------------
class PipelineLogger:
    @staticmethod
    def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level))

        if not logger.handlers:  
            handler = logging.FileHandler("pipeline.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            console = logging.StreamHandler()
            console.setFormatter(formatter)
            logger.addHandler(console)

        return logger


# --------------------------------------------------------
# PIPELINE
# --------------------------------------------------------
class MLPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger.setup_logger("MLPipeline", config.log_level)
        self.logger.info(f"Initialized with config {asdict(config)}")

        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}

    # -----------------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {filepath} shape={df.shape}")
        return df

    # -----------------------------
    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Preparing data")

        # Detect column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        # Preprocessing pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        # Full pipeline with model
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**self.config.model_params))
        ])

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # -----------------------------
    def train(self):
        self.logger.info("Training model")
        self.pipeline.fit(self.X_train, self.y_train)
        self.logger.info("Training finished")

    # -----------------------------
    def evaluate(self):
        self.logger.info("Evaluating model")
        y_pred = self.pipeline.predict(self.X_test)

        self.metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        }

        self.logger.info(f"Metrics: {self.metrics}")
        return self.metrics

    # -----------------------------
    def cross_validate(self):
        self.logger.info("Running cross-validation")

        scores = cross_val_score(
            self.pipeline,
            self.X_train,
            self.y_train,
            cv=self.config.cv_folds,
            scoring="accuracy"
        )

        cv_metrics = {
            "mean_cv_score": scores.mean(),
            "std_cv_score": scores.std()
        }

        self.logger.info(f"CV results: {cv_metrics}")
        return cv_metrics

    # -----------------------------
    def save_model(self, filepath: str):
        joblib.dump(self.pipeline, filepath)
        self.logger.info(f"Saved pipeline to {filepath}")

    def load_model(self, filepath: str):
        self.pipeline = joblib.load(filepath)
        self.logger.info(f"Loaded pipeline from {filepath}")

    # -----------------------------
    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)

    # -----------------------------
    def save_config(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(asdict(self.config), f, indent=4)
        self.logger.info(f"Saved config to {filepath}")


# --------------------------------------------------------
# USAGE
# --------------------------------------------------------
if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = MLPipeline(config)

    print("Pipeline created successfully!")
