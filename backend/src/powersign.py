import tensorflow as tf
import keras


# Implement PowerSign optimizer , discovered by the RNN made by [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)


class powerSign(keras.optimizers.Optimizer):
    def __init__(self, alpha=0.3, beta=0.5,  lr=0.5, epsilon=1e-7, useLocking=False, name="PowerSign"):
        super().__init__(name, useLocking)
        self.alpha = alpha
        self.beta = beta
        self.learningRate = lr
        self.epsilon = epsilon
        self._learning_rate = lr
        self.name = name
        self.m = None
        self._index_dict = {}
        self.slots = {}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "learningRate": self.learningRate,
                "epsilon": self.epsilon,
                "name": self.name

            }
        )

    def build(self, varList):
        self.slots = [tf.Variable(tf.zeros_like(var), trainable=True) for var in varList]

    def update_step(self, gradient, weight):
        learningRate = self.learningRate
        alpha = self.alpha
        beta = self.beta
        epsilon = tf.cast(self.epsilon, weight.dtype.base_dtype)

        for gradient, weight in zip(gradient, weight):
            m = self.slots[weight]
            mt = tf.maximum(beta*m+epsilon, tf.abs(gradient))
            weight.assign_sub(learningRate*gradient*tf.pow(alpha, tf.sign(mt)*tf.sign(gradient)))