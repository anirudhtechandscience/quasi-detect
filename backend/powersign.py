from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf


#Implement PowerSign optimizer , discovered by the RNN made by [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)


class PowerSign(tf.keras.optimizers.Optimizer):
    def __init__(self, alpha, beta, lr, useLocking=False, name="PowerSign"):
        super(PowerSign, self).__init__(useLocking, name)
        self.alpha = alpha
        self.beta = beta
        self.learningRate = lr

        self.tensorAlpha = None
        self.tensorBeta = None
        self.tensorLearningRate = None

    def _prepare(self):
        self.tensorAlpha = ops.convert_to_tensor(self.alpha, dtype=tf.float32)
        self.tensorBeta = ops.convert_to_tensor(self.beta, dtype=tf.float32)
        self.tensorLearningRate = ops.convert_to_tensor(self.learningRate, dtype=tf.float32)

    def _create_slots(self, varList):
        for _ in varList:
            self._zeros_slot(_, "m", self._name)

    def _apply_dense(self, gradient, weight):
        learningRate = math_ops.cast(self.tensorLearningRate, weight.dtype.base_dtype)
        alpha = math_ops.cast(self.tensorAlpha, weight.dtype.base_dtype)
        beta = math_ops.cast(self._beta_t, weight.dtype.base_dtype)

        m = self.get_slot(weight, "m")
        mt = m.assign(tf.maximum(beta * m, tf.abs(gradient)))
        update = state_ops.assign_sub(weight, learningRate * gradient * tf.pow(alpha, tf.sign(mt)*tf.sign(gradient)))
        return control_flow_ops.group(*[update, mt])

    def apply_sparse(self, gradient, weight):
        raise NotImplementedError("Currently my implementation of PowerSign doesn't have sparse gradient update,maybe i'll add it in the future ")
