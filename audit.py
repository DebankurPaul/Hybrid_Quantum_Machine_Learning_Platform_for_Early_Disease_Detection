import sys
import os
import json
from src import result_registry
from Streamlit_dashboard.dashboard_core import backend_adapter

print("=" * 60)
print("PHASE 0B REVISED AUDIT")
print("This script audits the registry using discovered datasets.")
print("=" * 60)

catalog = backend_adapter.get_dataset_catalog()
print("DATASETS IN REGISTRY:")
for d_key, d_name in catalog.items():
    print(f"- {d_key}: {d_name}")
    models = backend_adapter.get_available_models(d_key)
    print(f"  MODELS:")
    for m_key, m_name in models.items():
        caps = backend_adapter.get_capabilities(d_key, m_key)
        print(f"    - {m_key}: {m_name}")
        print(f"      Capabilities: {json.dumps(caps)}")
        
        meta = backend_adapter.get_model_metadata(d_key, m_key)
        if meta["status"] == "available":
            m_data = meta["data"]
            if m_data.get('protocol'):
                print(f"      Protocol: {json.dumps(m_data.get('protocol'))}")
            if m_data.get('preprocessing'):
                print(f"      Preprocessing: {json.dumps(m_data.get('preprocessing'))}")
            
            metrics = m_data.get('metrics', {})
            if metrics:
                print(f"      Metrics keys: {list(metrics.keys())}")
            if "quantum_config" in m_data and m_data["quantum_config"]:
                print(f"      Quantum Config: {json.dumps(m_data['quantum_config'])}")
        print("")
