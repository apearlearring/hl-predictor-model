import os
import tensorflow as tf

def setup_tensorflow():
    """Configure TensorFlow settings to minimize warnings and ensure consistent behavior"""
    # Disable all TensorFlow logging except errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Disable oneDNN optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Disable GPU to prevent additional warnings
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Enable deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Configure TensorFlow behavior
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.disable_eager_execution()
    
    # Configure Keras backend
    from tensorflow.keras import backend as K
    K.set_floatx('float32') 