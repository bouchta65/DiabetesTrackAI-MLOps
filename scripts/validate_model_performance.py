
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

def validate_model_performance():
    print("🤖 MODEL PERFORMANCE VALIDATION")
    print("="*60)
    
    # Thresholds
    MIN_ACCURACY = 0.70
    MIN_PRECISION = 0.65
    MIN_RECALL = 0.65
    MIN_F1 = 0.65
    MIN_ROC_AUC = 0.70
    
    errors = []
    
    # Find data
    data_path = 'data/processed/Ready_For_Model.csv'
    if not os.path.exists(data_path):
        data_path = 'data/raw/dataset-diabete-68e2810ab0d7e949117525.csv'
    
    if not os.path.exists(data_path):
        errors.append("❌ No data file found")
        print("\n".join(errors))
        sys.exit(1)
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Prepare data
        if 'Outcome' in df.columns:
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train model
        print("\n🏋️  Training validation model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n📊 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} (min: {MIN_ACCURACY})")
        print(f"  Precision: {precision:.4f} (min: {MIN_PRECISION})")
        print(f"  Recall:    {recall:.4f} (min: {MIN_RECALL})")
        print(f"  F1 Score:  {f1:.4f} (min: {MIN_F1})")
        print(f"  ROC AUC:   {roc_auc:.4f} (min: {MIN_ROC_AUC})")
        
        # Validate thresholds
        if accuracy < MIN_ACCURACY:
            errors.append(f"❌ Accuracy {accuracy:.4f} < {MIN_ACCURACY}")
        if precision < MIN_PRECISION:
            errors.append(f"❌ Precision {precision:.4f} < {MIN_PRECISION}")
        if recall < MIN_RECALL:
            errors.append(f"❌ Recall {recall:.4f} < {MIN_RECALL}")
        if f1 < MIN_F1:
            errors.append(f"❌ F1 {f1:.4f} < {MIN_F1}")
        if roc_auc < MIN_ROC_AUC:
            errors.append(f"❌ ROC AUC {roc_auc:.4f} < {MIN_ROC_AUC}")
        
    except Exception as e:
        errors.append(f"❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    if errors:
        print("❌ MODEL PERFORMANCE VALIDATION FAILED")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("✅ MODEL PERFORMANCE VALIDATION PASSED")
    print("="*60)

if __name__ == "__main__":
    validate_model_performance()

