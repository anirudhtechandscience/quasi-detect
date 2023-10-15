from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class ToBeImplementedError(Exception):

    def __init__(self,errormessage):
        super.__init__(errormessage)
class powerSign(tf.keras.optimizers.Optimizer):
    def __init__(self, alpha, beta, lr):
        super(powerSign, self).__init__(use_locking, name)
        self.alpha = alpha
        self.beta = beta
        self.learningRate = lr

        self.tensorAlpha = None
        self.tensorBeta = None
        self.tensorLearningRate = None
    def _prepare(self):
        self.tensorAlpha = ops.convert_to_tensor(self.alpha,dtype=tf.float32)
        self.tensorBeta = ops.convert_to_tensor(self.beta,dtype=tf.float32)
        self.tensorLearningRate = ops.convert_to_tensor(self.learningRate,dtype=tf.float32)


    def _create_slots(self,varList):
        for _ in varList:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self):
        raise ToBeImplementedError("Not implemented yet, excuse me")

    def apply_sparse(self):
        raise NotImplementedError("Currently my implementation of PowerSign doesn't have sparse gradient update,maybe i'll add it in the future ")