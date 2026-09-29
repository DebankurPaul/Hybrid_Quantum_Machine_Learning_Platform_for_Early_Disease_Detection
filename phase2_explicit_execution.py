import json
import pandas as pd
from src.inference import _run_registered_model, _validate_and_order_input, _normalise_labels
from src.result_registry import get_model_metadata
from src.data_service import get_authoritative_schema
import os

schema = get_authoritative_schema('wdbc')
features = schema.get('feature_names', [])
target_col = schema.get('target_column')
target_labels = _normalise_labels(schema)

raw_path = os.path.join('data', 'raw', 'wdbc', 'WDBC_raw.csv')
df = pd.read_csv(raw_path)
first_row = df.iloc[0]

input_data = {}
for f in features:
    if f in df.columns:
        input_data[f] = float(first_row[f])
    else:
        match = [c for c in df.columns if c.strip().replace(' ', '_').lower() == f.strip().replace(' ', '_').lower()]
        if match:
            input_data[f] = float(first_row[match[0]])
        else:
            input_data[f] = 0.0

print("\n=== EXPLICIT REAL DATA EXECUTION ===")
X_raw, err = _validate_and_order_input(input_data, features)
if err:
    print("Error in validation:", err)
    exit(1)

models = ["logistic_regression", "svm", "random_forest", "xgboost", "vqc"]
for model in models:
    meta_response = get_model_metadata("wdbc", model)
    model_meta = meta_response.get("data", {})
    model_meta["model_key"] = model
    
    result = _run_registered_model("wdbc", model_meta, X_raw, target_labels)
    print(f"\n--- WDBC {model} ---")
    print(json.dumps(result, indent=2, default=str))
