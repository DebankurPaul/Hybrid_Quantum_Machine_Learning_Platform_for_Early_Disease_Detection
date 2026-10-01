import json
import pandas as pd
import os

from src.data_service import get_authoritative_schema
from src.result_registry import get_model_metadata
from src.inference import _validate_and_order_input, _run_registered_model, _normalise_labels

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

    print("\n=== DIRECT VALIDATION ===")
    X_raw, error = _validate_and_order_input(input_data, features)
    if error:
        print("VALIDATION ERROR:", json.dumps(error, indent=2))
        return

    models = ["vqc", "logistic_regression", "svm", "random_forest", "xgboost"]
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

if __name__ == "__main__":
    run_tests()
