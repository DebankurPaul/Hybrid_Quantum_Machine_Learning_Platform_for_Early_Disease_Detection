import json
import sys
from src.inference import run_inference
from src.data_service import get_authoritative_schema

def print_result(title, result):
    print(f"\n--- {title} ---")
    print(json.dumps(result, indent=2, default=str))

print("=== 4. INFERENCE CONTRACT TESTS ===")
schema = get_authoritative_schema('wdbc')
if schema.get('status') != 'available':
    print("Failed to get WDBC schema")
    sys.exit(1)
features = schema.get('feature_names')
# Generate a dummy but valid real-like feature vector of zeroes
input_data = {f: 0.0 for f in features}

models = ['logistic_regression', 'svm', 'random_forest', 'xgboost', 'vqc']
for model in models:
    result = run_inference('wdbc', model, input_data)
    print_result(f"WDBC {model}", result)

print("\n=== 5. INPUT VALIDATION TESTS ===")
# Missing feature
missing_input = {f: 0.0 for f in features[1:]}
result_missing = run_inference('wdbc', 'logistic_regression', missing_input)
print_result("MISSING FEATURES TEST", result_missing)

# Unexpected feature
unexpected_input = {f: 0.0 for f in features}
unexpected_input['unexpected_feature_123'] = 0.0
result_unexpected = run_inference('wdbc', 'logistic_regression', unexpected_input)
print_result("UNEXPECTED FEATURES TEST", result_unexpected)

# Invalid numeric
invalid_input = {f: 0.0 for f in features}
invalid_input[features[0]] = 'not_a_number'
result_invalid = run_inference('wdbc', 'logistic_regression', invalid_input)
print_result("INVALID NUMERIC TEST", result_invalid)

print("\n=== 6. EVALUATION ONLY MODELS TESTS ===")
eval_models = [('uci', 'logistic_regression'), ('uci', 'qsvc'), ('golub', 'logistic_regression'), ('golub', 'qsvc')]
for dset, model in eval_models:
    # We can just pass empty dict because inference_ready check happens before validation
    res = run_inference(dset, model, {})
    print_result(f"{dset.upper()} {model}", res)
