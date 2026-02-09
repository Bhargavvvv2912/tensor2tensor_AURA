import os
import tensorflow as tf
import numpy as np
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

def smoke_test():
    print("Checking TensorFlow version...")
    print(f"TF Version: {tf.__version__}")
    
    # T2T v1.15.7 requires TF 1.x compatibility
    if tf.__version__.startswith('2.'):
        print("CRITICAL: TF 2.x detected. T2T requires TF 1.15 compatibility.")
        # Some legacy code might work with compat.v1, but T2T internals usually don't
    
    try:
        print("Attempting to initialize T2T Transformer Model...")
        # Create a tiny mock HParams set
        model_name = "transformer"
        hparams_set = "transformer_tiny"
        
        # Accessing registry to ensure T2T is properly hooked into TF
        hparams = registry.hparams(hparams_set)
        hparams.add_hparam("data_dir", "/tmp")
        
        # Verify model registration
        if model_name not in registry.list_models():
            raise ValueError(f"Model {model_name} not found in registry!")
            
        print("T2T Registry check passed.")
        
        # Basic sanity check: ensure tf_slim and other sub-deps are importable
        import tf_slim
        print("Sub-dependency 'tf_slim' check passed.")
        
        return True
    except Exception as e:
        print(f"SMOKE TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    if smoke_test():
        print("SUCCESS: T2T Environment is stable.")
        exit(0)
    else:
        exit(1)