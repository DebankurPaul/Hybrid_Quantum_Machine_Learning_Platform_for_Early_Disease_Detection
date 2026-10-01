import json
import pandas as pd
import os

from src.data_service import get_authoritative_schema
from src.result_registry import get_model_metadata
from src.inference import _validate_and_order_input, _run_registered_model, run_inference, _normalise_labels

def run_tests():
    # Load authoritative schema
    schema = get_authoritative_schema('wdbc')
    features = schema.get('feature_names', [])
    target_labels = _normalise_labels(schema)

    # Load actual first row
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

    print("=== ACTUAL WDBC ROW (first few keys) ===")
    print({k: input_data[k] for k in list(input_data.keys())[:5]})

    print("\n=== DIRECT VALIDATION ===")
    X_raw, error = _validate_and_order_input(input_data, features)
    if error:
        print("VALIDATION ERROR:", json.dumps(error, indent=2))
    else:
        print("VALIDATION SUCCESS, shape:", len(X_raw))

    models = ["logistic_regression", "svm", "random_forest", "xgboost", "vqc"]
    print("\n=== DIRECT EXECUTION ===")
    for model in models:
        meta_resp = get_model_metadata("wdbc", model)
        model_meta = dict(meta_resp.get("data", {}))
        model_meta["model_key"] = model
        
        try:
            result = _run_registered_model("wdbc", model_meta, X_raw, target_labels)
            print(f"\n--- WDBC {model.upper()} ---")
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"\n--- WDBC {model.upper()} FAILED ---")
            print(f"Exception: {type(e).__name__}: {str(e)}")

    print("\n=== PUBLIC API GATE TEST ===")
    for model in models:
        res = run_inference("wdbc", model, input_data)
        print(f"{model.upper()} PUBLIC API: {res.get('code')}")

    print("\n=== DIRECT INPUT VALIDATION ERROR TESTS ===")
    # Missing feature
    missing_input = input_data.copy()
    if len(features) > 0:
        del missing_input[features[0]]
    _, err_missing = _validate_and_order_input(missing_input, features)
    print("MISSING:", err_missing.get("code") if err_missing else "Passed unexpectedly")

    # Unexpected feature
    unexpected_input = input_data.copy()
    unexpected_input['unexpected_feature_999'] = 0.0
    _, err_unexpected = _validate_and_order_input(unexpected_input, features)
    print("UNEXPECTED:", err_unexpected.get("code") if err_unexpected else "Passed unexpectedly")

    # Invalid numeric
    invalid_input = input_data.copy()
    if len(features) > 0:
        invalid_input[features[0]] = 'not_a_number'
    _, err_invalid = _validate_and_order_input(invalid_input, features)
    print("INVALID NUMERIC:", err_invalid.get("code") if err_invalid else "Passed unexpectedly")

    print("\n=== EVALUATION ONLY (PUBLIC API) ===")
    eval_models = [('uci', 'logistic_regression'), ('uci', 'qsvc'), ('golub', 'logistic_regression'), ('golub', 'qsvc')]
    for dset, model in eval_models:
        res = run_inference(dset, model, input_data)
        print(f"{dset.upper()} {model.upper()}: {res.get('code')}")

if __name__ == "__main__":
    run_tests()
