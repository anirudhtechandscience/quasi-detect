from astropy.io import fits as fs 
import numpy as np
import os
import logging
import time
import tensorflow as tf

#This class uses standard Python conventions, such as _ to indicate private objects
#Also, I use CamelCase across the entire codebase, both frontend and backend




class loadData:

    def __init__(self,qsoPath="/home/tux/Downloads/QSO",
    othPath="/home/tux/Downloads/oth"):
        qsoSpectra = self._readSpectra(qsoPath)
        othSpectra = self._readSpectra(othPath)
        self.spectra = np.concatenate((qsoSpectra, othSpectra), axis=0)
        
        qsoLabels = np.ones(qsoSpectra.shape[0], dtype=np.int8)
        othLabels = np.zeros(othSpectra.shape[0], dtype=np.int8)
        self.labels = np.concatenate((qsoLabels, othLabels), axis=0)



    def _readSpectra(self, path):
        spectra = np.empty((0, 2, 5000), dtype=np.longdouble)        
        for fileName in os.listdir(path):
            print(spectra.shape)
            if fileName.endswith('.fits'):
                filePath= os.path.join(path, fileName)
                spectrum = self._getSpectrum(filePath)
                try:
                    spectra = np.append(spectra, [spectrum],axis=0)
                    logging.debug(f"Spectra shape:{spectra.shape}")
                except FileNotFoundError:
                    logging.critical("FileNotFoundError")
                except Exception as e:
                    logging.critical(f"A error occured reading {filePath}: {e}")
        logging.info(f"Final spectra shape:{spectra.shape}")
        time.sleep(15)
        return spectra

    def _getSpectrum(self, filePath):
        """ returns the spectrum from a .fits file """

        with fs.open(filePath) as spec:
            binHDU = spec[1].data
            loglam = np.array(binHDU['loglam'], dtype=np.longdouble)
            flux = np.array(binHDU['flux'], dtype=np.longdouble)

            #TODO: Implement a dynamic maximum pad length system
            loglam = np.pad(loglam,(0, 5000 - len(loglam)), mode='constant', constant_values=0)
            flux = np.pad(flux, (0,5000 - len(flux)), mode='constant', constant_values=0)

            spectrum = np.stack([loglam, flux])     
        return spectrum
        



    def getData(self):
        """
            A simple method to retrieve the data
        """
        return (self.spectra,self.labels)


class preProcessor:
    def __init__(self,spectra):
        self._spectra = spectra[0]
        self._labels = spectra[1]
        self._outSpectra = None
        self._normalizedSpectra = None
        self._testFeatures = None
        self._testLabels = None
        self._trainFeatures = None
        self._trainLabels = None
        self.trainData = None
        self.testData = None

    def _normalize(self):
        """
            Just, well normalizes the data between 0 and 1.
            """
        loglam = np.array(dtype=np.longdouble)
        flux = np.array(dtype=np.longdouble)
        for _ in range(self._spectra.shape[0]):
            loglam = np.stack([loglam,self._spectra[_][0]])
        for __ in range(self._spectra.shape[0]):
            flux = np.stack([flux,self._spectra[__][1]])
        normalizeFlux = np.linalg.norm(flux, 'fro')
        normalizeLoglam = np.linalg.norm(loglam, 'fro')
        self._normalizedSpectra = np.copy(self._spectra)
        for ___ in range(self._spectra.shape[0]):
            self._normalizedSpectra[___][1] = self._normalizedSpectra[___][1] / normalizeFlux
        
        for ____ in range(self._spectra.shape[0]):
            self._normalizedSpectra[____][0] = self._normalizedSpectra[____][0] / normalizeLoglam


    
    def _randomShifts(self):
        """
            Add some very small random shifts to the spectra,to simulate noise, 
            and also to augment the data.
        """

        shiftValues = np.array([-0.0001,-0.00009,-0.00008,-0.00007,-0.00006
        ,-0.00005,-0.00004,-0.00003,-0.00002,-0.00001
        ,0.00001, 0.00002,0.00003,0.00004,
        0.00005,0.00006,0.00007,0.00008
        ,0.00009,0.0001]
        ,dtype=np.longdouble) 
        self._outSpectra = np.copy(self._normalizedSpectra)
        numShifts = self._normalizedSpectra.shape[0] * 0.75
        numSubShift = 450 #every sub array will have this amount of values shifted
        self._outSpectra = np.copy(self._normalizedSpectra)
        for _ in range(numShifts):
            random2dSubarray = np.random.randint(0,self._normalizedSpectra.shape[0]) 
            twoDSubarray = np.copy(self._normalizedSpectra[random2dSubarray])
            for ___ in range(numSubShift):
                random1dSubarray = np.random.randint(0,self._normalizedSpectra.shape[1])
                randomValue = np.random.randint(0,self._normalizedSpectra.shape[2])
                shiftValue = np.random.choice(shiftValues)
                twoDSubarray[random1dSubarray][randomValue] += shiftValue
            self._outSpectra = np.append(self._outSpectra,[twoDSubarray],axis=0)
        
    def _trainTestSplit(self):
        last15Percent = int(self._outSpectra.shape[0] * 0.85)
        self._testFeatures = self._outSpectra[last15Percent:]
        self._testLabels = self._labels[last15Percent:]
        self._trainFeatures = self._outSpectra[:last15Percent]
        self._trainLabels = self._labels[:last15Percent]
    
    def _convToTensorflowObject(self):
        self.trainData = tf.data.Dataset.from_tensor_slices((self._trainFeatures,self._trainLabels))
        self.testData = tf.data.Dataset.from_tensor_slices((self._testFeatures,self._testLabels))
        batchSize = 1024
        bufferSize = 100

        self.trainData = self.trainData.shuffle(bufferSize).batch(batchSize)
        self.testData = self.testData.batch(batchSize)



    def getTrainData(self):
        if self.trainData:
            return self.trainData
        else:
            self._normalize()
            self._randomShifts()
            self._trainTestSplit()
            self._convToTensorflowObject()
            return self.trainData()

    def getTestData(self):
        if self.testData:
            return self.testData
        else:
            self._normalize()
            self._randomShifts()
            self._trainTestSplit()
            self._convToTensorflowObject()
            return self.testData()


