import pandas as pd
import numpy as np
import sys
import os

def validate_data_quality():
    errors = []
    warnings = []
    
    raw_data_path = 'data/raw/dataset-diabete-68e2810ab0d7e949117525.csv'
    if not os.path.exists(raw_data_path):
        errors.append(f"❌ Raw data file not found: {raw_data_path}")
    else:
        print(f"✅ Raw data file exists: {raw_data_path}")
        
        try:
            df = pd.read_csv(raw_data_path)
            print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check missing values
            missing = df.isnull().sum()
            total_missing = missing.sum()
            if total_missing > 0:
                warnings.append(f"⚠️  {total_missing} missing values found")
            else:
                print("✅ No missing values")
            
            # Check duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                warnings.append(f"⚠️  {duplicates} duplicate rows found")
            else:
                print("✅ No duplicates")
            
            # Check minimum rows
            if df.shape[0] < 100:
                errors.append(f"❌ Insufficient data: {df.shape[0]} rows (min 100)")
            else:
                print(f"✅ Sufficient data: {df.shape[0]} rows")
            
            # Check outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
                if outliers > 0:
                    warnings.append(f"⚠️  {outliers} outliers in {col}")
            
        except Exception as e:
            errors.append(f"❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    if warnings:
        print("⚠️  WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    
    if errors:
        print("❌ DATA QUALITY VALIDATION FAILED")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("✅ DATA QUALITY VALIDATION PASSED")
    print("="*60)

if __name__ == "__main__":
    validate_data_quality()