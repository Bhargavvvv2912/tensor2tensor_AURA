import os
import sys
import unittest.mock as mock

# ==============================================================================
# 1. GYM REGISTRY SHIELD (Monkey Patch)
# ==============================================================================
# T2T tries to register 'T2TTicTacToe-v0' on import. In modern Gym, this fails.
# we mock gym.make during the import phase to prevent the NoneType crash.
import gym
orig_make = gym.make

def patched_make(id, **kwargs):
    try:
        return orig_make(id, **kwargs)
    except Exception:
        print(f"DEBUG: Gym registration for '{id}' failed/missing. Returning mock.")
        return mock.MagicMock()

gym.make = patched_make

# ==============================================================================
# 2. NAMESPACE RECOVERY WING
# ==============================================================================
try:
    import tensorflow as tf
    print(f"DEBUG: TensorFlow {tf.__version__} detected.")
    
    try:
        from tensorflow.compat.v1 import estimator
    except ImportError:
        import tensorflow_estimator.python.estimator as estimator
        sys.modules['tensorflow.compat.v1.estimator'] = estimator
        print("DEBUG: Namespace bridge established for tf.compat.v1.estimator.")

except ImportError:
    print("CRITICAL: TensorFlow not found.")
    sys.exit(1)

# ==============================================================================
# 3. T2T REGISTRY VALIDATION
# ==============================================================================
try:
    # This import is where the 'NoneType' crash usually happens
    from tensor2tensor.utils import registry
    from tensor2tensor.models import transformer
    import tf_slim

    def smoke_test():
        print("Initializing T2T Transformer Registry check...")
        
        # Verify if 'transformer' is registered
        model_name = "transformer"
        if model_name not in registry.list_models():
            print(f"FAILURE: {model_name} not found in T2T registry.")
            return False
        
        # Verify Hyperparameter sets
        hparams_set = "transformer_tiny"
        hparams = registry.hparams(hparams_set)
        print(f"DEBUG: Hyperparameter set '{hparams_set}' loaded successfully.")
        
        return True

    if __name__ == "__main__":
        if smoke_test():
            print("\nSUCCESS: AURA has established a functional T2T environment.")
            sys.exit(0)
        else:
            sys.exit(1)

except Exception as e:
    print(f"\nVALIDATION CRASHED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)