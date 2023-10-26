from astropy.io import fits as fs 
import numpy as np
import os
import logging
import time
import tensorflow as tf

#This class uses standard Python conventions, such as _ to indicate private objects
#Also, I use CamelCase across the entire codebase, both frontend and backend


class loadData:

    def __init__(self, qsoPath="/home/tux/Downloads/QSO",
    othPath="/home/tux/Downloads/oth"):
        qsoSpectra = self._readSpectra(qsoPath)
        othSpectra = self._readSpectra(othPath)
        self.spectra = np.concatenate((qsoSpectra, othSpectra), axis=0)


        qsoLabels =  np.ones(qsoSpectra.shape[0], dtype=np.int8)
        othLabels = np.zeros(othSpectra.shape[0], dtype=np.int8)
        self.labels = np.concatenate((qsoLabels, othLabels), axis=0)

    def _readSpectra(self, path):
        spectra = np.empty((0, 2, 5000), dtype=np.float32)
        i = 0
        for fileName in os.listdir(path):
            i += 1
            if i == 35000:
                break
            #its just a cap , because my computer freeze up when i try more

            print(spectra.shape)
            if fileName.endswith('.fits'):
                filePath = os.path.join(path, fileName)
                spectrum = self._getSpectrum(filePath)
                try:
                    spectra = np.append(spectra, [spectrum], axis=0)
                    logging.debug(f"Spectra shape:{spectra.shape}")
                except FileNotFoundError:
                    logging.critical("FileNotFoundError")
                except Exception as e:
                    logging.critical(f"A error occured reading {filePath}: {e}")
        logging.info(f"Final spectra shape:{spectra.shape}")
        time.sleep(15)
        return spectra

    @staticmethod
    def _getSpectrum(filePath):
        """ returns the spectrum from a .fits file """

        with fs.open(filePath) as spec:
            binHDU = spec[1].data
            loglam = np.array(binHDU['loglam'], dtype=np.float32)
            flux = np.array(binHDU['flux'], dtype=np.float32)

            #TODO: Implement a dynamic maximum pad length system
            loglam = np.pad(loglam, (0, 5000 - len(loglam)), mode='constant', constant_values=0)
            flux = np.pad(flux, (0, 5000 - len(flux)), mode='constant', constant_values=0)

            spectrum = np.stack([loglam, flux])     
        return spectrum
        



    def getData(self):
        """
            A simple method to retrieve the data
        """
        return (self.spectra, self.labels)


class preProcessor:
    def __init__(self, spectra):
        self._spectra = spectra[0]
        self._labels = spectra[1]
        self.outSpectra = None
        self._normalizedSpectra = None


    def _normalize(self):
        """
            Just, well normalizes the data between 0 and 1.
            """
        loglam = np.empty((0))
        flux = np.empty((0))
        for _ in range(self._spectra.shape[0]):
            print(self._spectra[_][0].shape)
            loglam = np.concatenate((loglam, self._spectra[_][0]), axis=None)
        for __ in range(self._spectra.shape[0]):
            flux = np.concatenate((flux, self._spectra[__][1]),axis=None)
        normalizeFlux = np.linalg.norm(flux, 'fro')
        normalizeLoglam = np.linalg.norm(loglam, 'fro')
        self._normalizedSpectra = np.copy(self._spectra)
        for ___ in range(self._spectra.shape[0]):
            self._normalizedSpectra[___][1] = self._normalizedSpectra[___][1] / normalizeFlux
        
        for ____ in range(self._spectra.shape[0]):
            self._normalizedSpectra[____][0] = self._normalizedSpectra[____][0] / normalizeLoglam


    
    def _convToTensorflowObject(self):
        self.outSpectra = tf.data.Dataset.from_tensor_slices((self._normalizedSpectra, self._labels))


    def getData(self):
        if self.outSpectra:
            return (self.outSpectra,self._labels)
        else:
            self._normalize()
            self._convToTensorflowObject()
            return (self.outSpectra,self._labels)


