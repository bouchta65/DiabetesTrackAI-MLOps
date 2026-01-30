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
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"\n📊 Class distribution:")
        print(class_counts)
        
        # Check if we have enough samples for stratified split
        min_class_count = class_counts.min()
        use_stratify = min_class_count >= 2
        
        if not use_stratify:
            print(f"⚠️  Warning: Smallest class has only {min_class_count} samples - skipping stratification")
        
        # Split - use stratify only if we have enough samples
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Check if we have samples from both classes in test set
        test_classes = y_test.nunique()
        if test_classes < 2:
            print("⚠️  Warning: Test set doesn't contain both classes - adjusting split...")
            # Use larger test size to ensure both classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            test_classes = y_test.nunique()
            
            if test_classes < 2:
                errors.append("❌ Insufficient data: Cannot create valid train/test split with both classes")
                raise ValueError("Cannot create balanced split")
        
        # Train model
        print("\n🏋️  Training validation model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Check if we have probability for positive class
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_positive = y_pred_proba[:, 1]
        else:
            y_pred_proba_positive = y_pred_proba[:, 0]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC only if we have both classes
        if y_test.nunique() == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
        else:
            roc_auc = 0.0
            print("⚠️  Warning: Cannot calculate ROC AUC (need both classes in test set)")
        
        print("\n📊 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} (min: {MIN_ACCURACY})")
        print(f"  Precision: {precision:.4f} (min: {MIN_PRECISION})")
        print(f"  Recall:    {recall:.4f} (min: {MIN_RECALL})")
        print(f"  F1 Score:  {f1:.4f} (min: {MIN_F1})")
        print(f"  ROC AUC:   {roc_auc:.4f} (min: {MIN_ROC_AUC})")
        
        # Validate thresholds
        if accuracy < MIN_ACCURACY:
            errors.append(f"❌ Accuracy {accuracy:.4f} < {MIN_ACCURACY}")
        else:
            print(f"✅ Accuracy meets threshold")
            
        if precision < MIN_PRECISION:
            errors.append(f"❌ Precision {precision:.4f} < {MIN_PRECISION}")
        else:
            print(f"✅ Precision meets threshold")
            
        if recall < MIN_RECALL:
            errors.append(f"❌ Recall {recall:.4f} < {MIN_RECALL}")
        else:
            print(f"✅ Recall meets threshold")
            
        if f1 < MIN_F1:
            errors.append(f"❌ F1 {f1:.4f} < {MIN_F1}")
        else:
            print(f"✅ F1 meets threshold")
            
        if y_test.nunique() == 2 and roc_auc < MIN_ROC_AUC:
            errors.append(f"❌ ROC AUC {roc_auc:.4f} < {MIN_ROC_AUC}")
        elif y_test.nunique() == 2:
            print(f"✅ ROC AUC meets threshold")
        
    except Exception as e:
        errors.append(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
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