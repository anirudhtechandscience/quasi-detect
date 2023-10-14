import tensorflow as tf
import tf.keras as keras
class powerSign(keras.optimizers.Optimizer):
    def __init__(self):
        self.alpha = None
        self.beta = None
