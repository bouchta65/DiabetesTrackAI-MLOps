import glob
import pandas as pd
import sys
from pathlib import Path
from data_validator import DataValidator

data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "*.csv"
validator = DataValidator()
csv_files = glob.glob(str(data_path))

if not csv_files:
    print("No CSV files found")
    sys.exit(1)

for file in csv_files:
    try:
        df = pd.read_csv(file)
        validator.validate_data(df)
        print(f"{Path(file).name}: OK")
    except Exception as e:
        print(f"{Path(file).name}: FAILED - {e}")
        sys.exit(1)

print(f"Data validation complete: {len(csv_files)} files passed")
