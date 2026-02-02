import glob
import pandas as pd
import sys
from pathlib import Path
from model_validator import ModelValidator

data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "*.csv"
validator = ModelValidator()
csv_files = glob.glob(str(data_path))

model_files = []
for file in csv_files:
    try:
        df = pd.read_csv(file, nrows=1)
        if 'Cluster' in df.columns:
            model_files.append(file)
    except:
        continue

if not model_files:
    print("No files with Cluster column found")
    sys.exit(1)

for file in model_files:
    try:
        df = pd.read_csv(file)
        X = df.drop("Cluster", axis=1)
        validator.validate_model_input(X)
        print(f"{Path(file).name}: OK")
    except Exception as e:
        print(f"{Path(file).name}: FAILED - {e}")
        sys.exit(1)

print(f"Model validation complete: {len(model_files)} files passed")
