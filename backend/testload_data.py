from load_data import loadData, preProcessor
ld = loadData(qsoPath="test/testQSO", othPath="test/testoth")



def testShape():
    assert ld.spectra.shape == (8, 2, 5000)


def testLabels():
    assert ld.labels[1] == 1
    assert ld.labels[-1] == 0
    assert ld.labels.shape == (8,)

def


