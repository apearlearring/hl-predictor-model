import os
import warnings
import tensorflow as tf

def setup_tensorflow():
    """Configure TensorFlow settings to minimize warnings and ensure consistent behavior"""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Disable all TensorFlow logging except errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Disable oneDNN optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Disable GPU to prevent additional warnings
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Enable deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Configure TensorFlow behavior using newer API
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Configure Keras backend
    tf.keras.backend.set_floatx('float32')