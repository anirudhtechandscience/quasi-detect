from backend.src.powersign import powerSign
import tensorflow as tf
import numpy as np


ps = powerSign(0.1, 0.2, 0.01, s1e-5)


def testVarInitializing():
    assert ps.alpha == 0.1
    assert ps.beta == 0.2
    assert ps.learningRate == 0.01
    assert ps.epsilon == 1e-5


def testTensorVar():
    ps._prepare()
    assert ps.tensorAlpha == 0.1
    assert ps.tensorBeta == 0.2
    assert ps.tensorLearningRate == 0.01
    assert ps.tensorEpsilon == 1e-5


def testIsTensor():
    ps._prepare()
    assert tf.is_tensor(ps.tensorAlpha)
    assert tf.is_tensor(ps.tensorBeta)
    assert tf.is_tensor(ps.tensorLearningRate)
    assert tf.is_tensor(ps.tensorEpsilon)


def testApplyDense():
    testGradient = np.array([2.89, 3.76], dtype=np.float16)
    varlist = np.array([3.69, 4.20], dtype=np.float16)
    ps._create_slots(varList=varlist)
    ps._apply_dense(testGradient, 3.69)
