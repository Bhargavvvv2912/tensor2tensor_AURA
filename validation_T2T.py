import os
import sys

# ==============================================================================
# NAMESPACE RECOVERY WING
# ==============================================================================
try:
    import tensorflow as tf
    print(f"DEBUG: TensorFlow {tf.__version__} detected.")
    
    # In TF 2.16+, 'tf.compat.v1.estimator' is often missing from the core.
    # We manually bridge it from the 'tensorflow_estimator' package.
    try:
        from tensorflow.compat.v1 import estimator
    except ImportError:
        print("DEBUG: Manual bridge required for tf.compat.v1.estimator...")
        import tensorflow_estimator.python.estimator as estimator
        # Inject it into the sys.modules so tensor2tensor can find it
        sys.modules['tensorflow.compat.v1.estimator'] = estimator
        print("DEBUG: Namespace bridge established.")

except ImportError:
    print("CRITICAL: TensorFlow not found in environment.")
    sys.exit(1)

# ==============================================================================
# T2T REGISTRY VALIDATION
# ==============================================================================
try:
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
        
        # Verify TF Slim integration (a common point of failure for T2T)
        print(f"DEBUG: tf_slim version: {tf_slim.__version__}")
        
        return True

    if __name__ == "__main__":
        if smoke_test():
            print("\nSUCCESS: AURA has established a functional T2T environment.")
            sys.exit(0)
        else:
            sys.exit(1)

except Exception as e:
    print(f"\nVALIDATION CRASHED: {e}")
    # Print the traceback to help the AURA agent diagnose the fix
    import traceback
    traceback.print_exc()
    sys.exit(1)