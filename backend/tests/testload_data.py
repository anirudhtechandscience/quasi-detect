import numpy as np
import tensorflow as tf
from backend.src.load_data import loadData, preProcessor
# TODO : Fix the relative paths, for the moment you'll have to plug the absolute paths in yourself
ld = loadData(qsoPath="/home/tux/quasi-detect/backend/src/test/testQSO", othPath="/home/tux/quasi-detect/backend/src/test/testoth")


# test the loadData class

def testShape():
    assert ld.spectra.shape == (8, 2, 5000)


def testLabels():
    assert ld.labels[1] == 1
    assert ld.labels[-1] == 0
    assert ld.labels.shape == (8,)


def testgetData():
    data = ld.getData()
    assert data[0].all() == ld.spectra.all()
    assert data[1].all() == ld.labels.all()


#test the preProcessor class

pp = preProcessor(ld.getData())


def testNormalize():
    pp._normalize()
    assert -0.3 <= np.min(pp._normalizedSpectra) <= 0.3
    assert np.max(pp._normalizedSpectra) < 1


def testTensorflowObject():
    data = pp.getData()
    assert isinstance(data, tf.data.Dataset)
