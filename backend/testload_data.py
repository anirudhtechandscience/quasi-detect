from load_data import loadData, preProcessor
ld = loadData(qsoPath="test/testQSO", othPath="test/testoth")


class testloadData:
    @staticmethod
    def testShape():
        assert ld.spectra.shape == (8, 2, 5000)

    @staticmethod
    def test():
        pass



