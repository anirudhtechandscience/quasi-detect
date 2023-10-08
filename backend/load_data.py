from astropy.io import fits as fs 
import numpy as np
import os


#This class uses standard Python conventions, such as _ to indicate private objects
#Also, I use CamelCase across the entire codebase, both frontend and backend


class loadData:

    def __init__(self,qsoPath="/home/tux/Downloads/QSO",othPath="/home/tux/Downloads/oth"):
        self._dirPathQuasar = qsoPath
        self._dirPathOther = othPath
        self._qsoSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self._othSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self._othLabels = None
        self._qsoLabels = None
        self.labels = None
        self.spectra = None
        
    def loadQSOData(self):
        """
            This loads the relevant columns from the .fits files , which is basically 
            the spectrum, and then makes a array with a 1 for every spectrum, as the 
            labels(Initially it will just tell quasars apart from other objects to 
            reduce complexity, later it will detect all kinds of objects). Both arrays 
            will be combined with the arrays of the stars and galaxies using 
            the combineData methods.
        """ 
        try:
            for fileName in os.listdir(self._dirPathQuasar):
                if fileName.endswith('.fits'):
                    filePath = os.path.join(self._dirPathQuasar, fileName)
                    with fs.open(filePath) as spec:
                        binHDU = spec[1].data
                        loglam = np.array(binHDU['loglam'],dtype=np.longdouble)
                        flux = np.array(binHDU['flux'],dtype=np.longdouble)
                        #TODO: Implement a dynamic maximum pad length system
                        loglam = np.pad(loglam, (0, 5000 - len(loglam)), mode='constant', constant_values=0)
                        flux = np.pad(flux, (0,5000 - len(flux)), mode='constant', constant_values=0)
                        spectrum = np.stack([loglam,flux])
                        self._qsoSpectra = np.append(self._qsoSpectra, [spectrum], axis=0)
            self._qsoLabels = np.ones(self._qsoSpectra.shape[0],dtype=np.int8)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")


    def loadOthData(self):
        """
            This loads the relevant columns from the .fits files , which is basically 
            the spectrum, and then makes a array with a 0 for every spectrum, as the 
            labels(Initially it will just tell quasars apart from other objects to 
            reduce complexity, later it will detect all kinds of objects). Both arrays 
            will be combined with the arrays of the quasars using 
            the combineData methods.
        """ 
        try:
            for fileName in os.listdir(self._dirPathOther):
                if fileName.endswith('.fits'):
                    filePath = os.path.join(self._dirPathOther, fileName)
                    with fs.open(filePath) as spec:
                        binHDU = spec[1].data
                        loglam = np.array(binHDU['loglam'],dtype=np.longdouble)
                        flux = np.array(binHDU['flux'],dtype=np.longdouble)
                        #TODO: Implement a dynamic maximum pad length system
                        loglam = np.pad(loglam, (0, 5000 - len(loglam)), mode='constant', constant_values=0)
                        flux = np.pad(flux, (0,5000 - len(flux)), mode='constant', constant_values=0)
                        spectrum = np.stack([loglam,flux])
                        self._othSpectra = np.append(self._othSpectra, [spectrum], axis=0)
            self._othLabels = np.zeros(self._othSpectra.shape[0],dtype=np.int8)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")

    def combineDataLabels(self):
        """
            This combines both the arrays of labels into one array, to be fed 
            into the neural network.
        """

        self.labels = np.concatenate((self._qsoLabels,self._othLabels),axis=0)

    def combineDataSpectra(self):
        """
            This combines both the arrays of labels into one array, to be fed 
            into the preProcessor class
        """
        self.spectra = np.concatenate((self._qsoSpectra,self._othSpectra),axis=0)

    @staticmethod
    def getSpectra(self):
        """
            A simple method to retrieve the spectra
        """
        if self.spectra:
            return self.spectra
        else:
            self.loadQSOData()
            self.loadOthData()
            self.combineDataSpectra()
            return self.spectra

    @staticmethod
    def getLabels(self):
        """
            A simple method to retrieve the labels
        """
        if self.labels:
            return self.labels
        else:
            self.loadQSOData()
            self.loadOthData()
            self.combineDataLabels()
            return self.labels


dataLoader = loadData()

class preProcessor:
    def __init__(self,spectra=dataLoader.getSpectra):
        self.spectra = spectra
        self.outSpectra = None



