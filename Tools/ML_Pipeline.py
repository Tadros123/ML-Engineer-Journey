import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    test_size: float = 0.2
    random_state: int = 42
    scaler_type: str = "standard"
    model_type: str = "random_forest"
    model_params: Dict[str, Any] = None
    cv_folds: int = 5
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": self.random_state
            }


class PipelineLogger:
    """Centralized logging for ML pipeline"""
    
    @staticmethod
    def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level))
        
        handler = logging.FileHandler("pipeline.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger


class MLPipeline:
    """Reusable ML classification pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger.setup_logger(
            "MLPipeline", 
            config.log_level
        )
        self.logger.info(f"Initialized pipeline with config: {asdict(config)}")
        
        self.scaler = StandardScaler()
        self.model = self._initialize_model()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
    
    def _initialize_model(self):
        """Initialize model based on config"""
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(**self.config.model_params)
        else:
            self.logger.error(f"Unknown model type: {self.config.model_type}")
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV"""
        try:
            data = pd.read_csv(filepath)
            self.logger.info(f"Loaded data from {filepath}. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
    
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split and scale data"""
        self.logger.info("Starting data preparation")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        self.logger.info(
            f"Data split - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
        )
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self):
        """Train the model"""
        if self.X_train is None:
            self.logger.error("Data not prepared. Call prepare_data() first.")
            raise ValueError("Data not prepared")
        
        self.logger.info("Starting model training")
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Model training completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        if self.X_test is None:
            self.logger.error("Data not prepared. Call prepare_data() first.")
            raise ValueError("Data not prepared")
        
        self.logger.info("Evaluating model")
        y_pred = self.model.predict(self.X_test)
        
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average="weighted"),
            "recall": recall_score(self.y_test, y_pred, average="weighted"),
            "f1": f1_score(self.y_test, y_pred, average="weighted")
        }
        
        self.metrics = metrics
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(self) -> Dict[str, float]:
        """Perform cross-validation"""
        if self.X_train is None:
            self.logger.error("Data not prepared. Call prepare_data() first.")
            raise ValueError("Data not prepared")
        
        self.logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")
        scores = cross_val_score(
            self.model, 
            self.X_train, 
            self.y_train,
            cv=self.config.cv_folds,
            scoring="accuracy"
        )
        
        cv_metrics = {
            "mean_cv_score": scores.mean(),
            "std_cv_score": scores.std()
        }
        self.logger.info(f"Cross-validation scores: {cv_metrics}")
        return cv_metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_config(self, filepath: str):
        """Save configuration to JSON"""
        try:
            config_dict = asdict(self.config)
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.info(f"Config saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        test_size=0.2,
        random_state=42,
        log_level="INFO"
    )
    
    pipeline = MLPipeline(config)
    print("Pipeline created successfully!")