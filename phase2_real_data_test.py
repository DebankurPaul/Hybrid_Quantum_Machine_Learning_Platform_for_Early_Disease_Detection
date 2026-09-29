import json
import pandas as pd
from src.inference import run_inference, _validate_and_order_input
from src.data_service import get_authoritative_schema
import os

schema = get_authoritative_schema('wdbc')
features = schema.get('feature_names', [])
target_col = schema.get('target_column')
target_labels = schema.get('target_labels')

# 1 & 2: Obtain actual data row
raw_path = os.path.join('data', 'raw', 'wdbc', 'WDBC_raw.csv')
if not os.path.exists(raw_path):
    print("ERROR: raw data not found at", raw_path)
    exit(1)

df = pd.read_csv(raw_path)
first_row = df.iloc[0]

# 3: Construct input_data
input_data = {}
for f in features:
    if f in df.columns:
        input_data[f] = float(first_row[f])
    else:
        # try matching without spaces or lowercasing if there's a mismatch
        match = [c for c in df.columns if c.strip().replace(' ', '_').lower() == f.strip().replace(' ', '_').lower()]
        if match:
            input_data[f] = float(first_row[match[0]])
        else:
            print(f"Feature {f} not found in raw CSV")
            input_data[f] = 0.0

print(f"\n--- SOURCE ROW INFO ---")
actual_target_value = first_row.get(target_col, "UNKNOWN")
if isinstance(actual_target_value, pd.Series):
    actual_target_value = actual_target_value.iloc[0]
print(f"Target Column: {target_col}")
print(f"Actual Target Value: {actual_target_value}")

print("\n=== 4. REAL DATA INFERENCE ===")
models = ["logistic_regression", "svm", "random_forest", "xgboost", "vqc"]
for model in models:
    result = run_inference("wdbc", model, input_data)
    print(f"\n--- WDBC {model} ---")
    print(json.dumps(result, indent=2, default=str))

print("\n=== 8. INPUT VALIDATION TESTS ===")
# Missing feature
missing_input = input_data.copy()
if len(features) > 0:
    del missing_input[features[0]]
result_missing = run_inference('wdbc', 'logistic_regression', missing_input)
if result_missing.get("code") == "INFERENCE_NOT_READY":
    _, result_missing = _validate_and_order_input(missing_input, features)
print("\n--- MISSING FEATURES TEST ---")
print(json.dumps(result_missing, indent=2, default=str))

# Unexpected feature
unexpected_input = input_data.copy()
unexpected_input['unexpected_feature_999'] = 0.0
result_unexpected = run_inference('wdbc', 'logistic_regression', unexpected_input)
if result_unexpected.get("code") == "INFERENCE_NOT_READY":
    _, result_unexpected = _validate_and_order_input(unexpected_input, features)
print("\n--- UNEXPECTED FEATURES TEST ---")
print(json.dumps(result_unexpected, indent=2, default=str))

# Invalid numeric
invalid_input = input_data.copy()
if len(features) > 0:
    invalid_input[features[0]] = 'not_a_number'
result_invalid = run_inference('wdbc', 'logistic_regression', invalid_input)
if result_invalid.get("code") == "INFERENCE_NOT_READY":
    _, result_invalid = _validate_and_order_input(invalid_input, features)
print("\n--- INVALID NUMERIC TEST ---")
print(json.dumps(result_invalid, indent=2, default=str))

print("\n=== 9 & 10. EVALUATION ONLY MODELS TESTS ===")
eval_models = [('uci', 'logistic_regression'), ('uci', 'qsvc'), ('golub', 'logistic_regression'), ('golub', 'qsvc')]
for dset, model in eval_models:
    res = run_inference(dset, model, {})
    print(f"\n--- {dset.upper()} {model} ---")
    print(json.dumps(res, indent=2, default=str))
