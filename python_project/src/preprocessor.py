import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def validate_data(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if X.isnull().any().any():
            logger.warning("Input contains missing values. They will be filled with mean values.")
            X.fillna(X.mean(), inplace=True)

    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        self.validate_data(X)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(X)
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        try:
            transformed_data = self.scaler.transform(X)
            return pd.DataFrame(transformed_data, columns=X.columns)
        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            raise

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, filepath: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        try:
            joblib.dump(self, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise

    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        try:
            preprocessor = joblib.load(filepath)
            logger.info(f"Preprocessor loaded from {filepath}")
            return preprocessor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise