import json
from src.inference import _validate_and_order_input
from src.data_service import get_authoritative_schema

schema = get_authoritative_schema('wdbc')
features = schema.get('feature_names')

print("--- MISSING FEATURES TEST ---")
missing_input = {f: 0.0 for f in features[1:]}
_, err = _validate_and_order_input(missing_input, features)
print(json.dumps(err, indent=2))

print("--- UNEXPECTED FEATURES TEST ---")
unexpected_input = {f: 0.0 for f in features}
unexpected_input['unexpected_feature_123'] = 0.0
_, err = _validate_and_order_input(unexpected_input, features)
print(json.dumps(err, indent=2))

print("--- INVALID NUMERIC TEST ---")
invalid_input = {f: 0.0 for f in features}
invalid_input[features[0]] = 'not_a_number'
_, err = _validate_and_order_input(invalid_input, features)
print(json.dumps(err, indent=2))
