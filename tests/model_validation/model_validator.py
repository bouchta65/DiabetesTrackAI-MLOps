import pandas as pd
import numpy as np

class ModelValidator:
    def __init__(self, expected_features=8):
        self.expected_features = expected_features
        self.expected_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

    def validate_input_shape(self, X):
        """Validate input data shape"""
        if hasattr(X, 'shape'):
            if len(X.shape) != 2:
                raise ValueError(f"Expected 2D array, got {len(X.shape)}D")
            if X.shape[1] != self.expected_features:
                raise ValueError(f"Expected {self.expected_features} features, got {X.shape[1]}")

    def validate_input_types(self, X):
        """Validate input data types"""
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    raise ValueError(f"Column {col} must be numeric")
        elif hasattr(X, 'dtype'):
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("All features must be numeric")

    def validate_input_columns(self, X):
        """Validate DataFrame columns match expected"""
        if isinstance(X, pd.DataFrame):
            missing = [col for col in self.expected_columns if col not in X.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

    def validate_model_input(self, X):
        """Complete model input validation"""
        self.validate_input_shape(X)
        self.validate_input_types(X)
        if isinstance(X, pd.DataFrame):
            self.validate_input_columns(X)