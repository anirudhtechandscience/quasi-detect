from load_data import loadData, preProcessor
import numpy as np
#simple script to run all the other scipts

dataLoader = loadData()
data = dataLoader.getData()

process = preProcessor(spectra=data)
preProcessedData = process.getData()
preprocessedSpec = preProcessedData[0]
print(preProcessedSpec[2])
print(np.min(preProcessedSpec))
print(np.max(preProcessedSpec))

np.savetxt("features.csv", preProcessedData[0], delimiter=",")
np.savetxt("labels.csv", preProcessedData[1], delimiter=",")

#Here I just save all
# the data to slightly optimize over my for loop when  data is loaded, I will release it when I get around to running it.
