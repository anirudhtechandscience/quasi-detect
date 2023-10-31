from backend.src.powersign import powerSign


ps = powerSign(alpha=0.1, beta=0.2, lr=0.01,epsilon=1e-5)


def testVarInitializing():
    assert ps.alpha == 0.1
    assert ps.beta == 0.2
    assert ps.learningRate == 0.01
    assert ps.epsilon == 1e-5

def testTensorvar():
    ps._prepare()
    assert ps.tensorAlpha == 0.1
    assert ps.tensorBeta == 0.2
    assert ps.tensorLearningRate == 0.01
    assert ps.tensorEpsilon == 1e-5

