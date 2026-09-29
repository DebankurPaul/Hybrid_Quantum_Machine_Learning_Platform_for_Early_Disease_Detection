import json
import sys

from src import result_registry
from Streamlit_dashboard.dashboard_core import backend_adapter as backend

print("=== REGISTRY DISCOVERY CHECK ===")
datasets = result_registry.get_datasets()
print("Discovered Datasets:", datasets)

for dkey, dname in datasets.items():
    print(f"\nDataset: {dkey} ({dname})")
    models = result_registry.get_models(dkey)
    print("  Models:", models)
    
    for mkey, mname in models.items():
        print(f"  Model: {mkey} ({mname})")
        
        caps = result_registry.get_capabilities(dkey, mkey)
        print("    Capabilities:", caps)
        
        metrics = result_registry.get_metrics(dkey, mkey)
        print("    Metrics Status:", metrics.get("status"))
        
        schema = result_registry.get_prediction_schema(dkey, mkey)
        print("    Schema Status:", schema.get("status"))
        
        model_type = result_registry.get_model(dkey, mkey).get("data", {}).get("type")
        if model_type == "Quantum":
            qc = result_registry.get_quantum_configuration(dkey, mkey)
            print("    Quantum Config Status:", qc.get("status"))
            
        expl = result_registry.get_explainability(dkey, mkey)
        print("    Explainability Status:", expl.get("status"))

print("\n=== COMPATIBILITY CHECK ===")
try:
    c = backend.get_dataset_catalog()
    print("get_dataset_catalog OK")
    
    for dkey in datasets.keys():
        backend.get_available_models(dkey)
        backend.get_dataset_metadata(dkey)
        backend.get_data_sample(dkey)
        for mkey in result_registry.get_models(dkey).keys():
            backend.get_capabilities(dkey, mkey)
            backend.get_model_metadata(dkey, mkey)
    print("Compatibility functions OK")
except Exception as e:
    print("Compatibility Check FAILED:", str(e))
    sys.exit(1)
