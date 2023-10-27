from load_data import loadData, preProcessor
import numpy as np
#simple script to run all the other scipts

anwser = input("Do you want to train the model or run the tests? Type train or test")
if anwser == "train":
    dataLoader = loadData()
    data = dataLoader.getData()

    process = preProcessor(spectra=data)
    preProcessedData = process.getData()
    preprocessedSpec = preProcessedData[0]
else:


