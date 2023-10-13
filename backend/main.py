from load_data import loadData, preProcessor
import numpy as np
#simple script to run all the other scipts

dataLoader = loadData()
data = dataLoader.getData()

process = preProcessor(spectra=data)
preProcessedData = process.getData()
preprocessedSpec = preProcessedData[0]
print(preProcessedData[2])
print(np.min(preProcessedData))
print(np.max(preProcessedData))

np.savetxt("features.csv",preProcessedData[0],delimiter=",")
np.savetxt("labels.csv",preProcessedData[1],delimiter=",")