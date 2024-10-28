# data_preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame):
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

    def save(self, filepath: str):
        import joblib
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        import joblib
        return joblib.load(filepath)
