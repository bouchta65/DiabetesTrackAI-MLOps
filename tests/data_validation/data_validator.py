import pandas as pd

class DataValidator:
    def __init__(self):
        self.required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.feature_ranges = {
            'Pregnancies': (0, 30),
            'Glucose': (0, 300),
            'BloodPressure': (0, 200),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (0, 70),  
            'DiabetesPedigreeFunction': (0, 3),
            'Age': (0, 120)
        }

    def validate_columns(self, df):
        """Check all required columns exist"""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def validate_numeric_types(self, df):
        """Check numeric columns are numeric"""
        for col in self.required_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

    def validate_missing_values(self, df, fill_missing=False):
        """Check for missing values"""
        if df.isnull().any().any():
            if fill_missing:
                return df.fillna(df.median())
            else:
                raise ValueError("Dataset contains missing values")
        return df

    def validate_ranges(self, df):
        """Validate feature ranges"""
        for col, (min_val, max_val) in self.feature_ranges.items():
            if col in df.columns:
                if (df[col] < min_val).any() or (df[col] > max_val).any():
                    raise ValueError(f"Column {col} values outside range [{min_val}, {max_val}]")

    def validate_data(self, df, fill_missing=False):
        """Complete data validation"""
        self.validate_columns(df)
        self.validate_numeric_types(df)
        df = self.validate_missing_values(df, fill_missing)
        self.validate_ranges(df)
        return df