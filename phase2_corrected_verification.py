import json
import pandas as pd
import os
from src.inference import run_inference
from src.data_service import get_authoritative_schema

schema = get_authoritative_schema('wdbc')
features = schema.get('feature_names', [])
target_col = schema.get('target_column')

raw_path = os.path.join('data', 'raw', 'wdbc', 'WDBC_raw.csv')
if not os.path.exists(raw_path):
    print("ERROR: raw data not found at", raw_path)
    exit(1)

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

print("\n=== PUBLIC API REAL DATA INFERENCE ===")
models = ["logistic_regression", "svm", "random_forest", "xgboost", "vqc"]
for model in models:
    result = run_inference("wdbc", model, input_data)
    print(f"\n--- WDBC {model} ---")
    print(json.dumps(result, indent=2, default=str))

print("\n=== INPUT VALIDATION TESTS ===")
# Missing feature
missing_input = input_data.copy()
if len(features) > 0:
    del missing_input[features[0]]
result_missing = run_inference('wdbc', 'logistic_regression', missing_input)
print("\n--- MISSING FEATURES TEST ---")
print(json.dumps(result_missing, indent=2, default=str))

# Unexpected feature
unexpected_input = input_data.copy()
unexpected_input['unexpected_feature_999'] = 0.0
result_unexpected = run_inference('wdbc', 'logistic_regression', unexpected_input)
print("\n--- UNEXPECTED FEATURES TEST ---")
print(json.dumps(result_unexpected, indent=2, default=str))

# Invalid numeric
invalid_input = input_data.copy()
if len(features) > 0:
    invalid_input[features[0]] = 'not_a_number'
result_invalid = run_inference('wdbc', 'logistic_regression', invalid_input)
print("\n--- INVALID NUMERIC TEST ---")
print(json.dumps(result_invalid, indent=2, default=str))

print("\n=== EVALUATION ONLY MODELS TESTS ===")
eval_models = [('uci', 'logistic_regression'), ('uci', 'qsvc'), ('golub', 'logistic_regression'), ('golub', 'qsvc')]
for dset, model in eval_models:
    res = run_inference(dset, model, input_data)
    print(f"\n--- {dset.upper()} {model} ---")
    print(json.dumps(res, indent=2, default=str))
