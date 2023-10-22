import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as pyplot
from powersign import powerSign


class Train:

    def __init__(self, traindata, testdata):
        self.train = traindata
        self.test = testdata

    def neural_network(self):
