from astropy.io import fits as fs 
import numpy as np
import os


#This class uses standard Python conventions, such as _ to indicate private objects
#Also, I use CamelCase across the entire codebase, both frontend and backend


class loadData:

    def __init__(self,qsoPath="/home/tux/Downloads/QSO",
    othPath="/home/tux/Downloads/oth"):

        self._dirPathQuasar = qsoPath
        self._dirPathOther = othPath
        self._qsoSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self._othSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self._othLabels = None
        self._qsoLabels = None
        self.labels = None
        self.spectra = None
        self._maxLength = 5000
        
    def _loadQSOData(self):
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
                        loglam = np.pad(loglam, (0, self._maxLength - len(loglam))
                        , mode='constant', constant_values=0)

                        flux = np.pad(flux, (0,self._maxLength - len(flux))
                        , mode='constant', constant_values=0)

                        spectrum = np.stack([loglam,flux])
                        self._qsoSpectra = np.append(self._qsoSpectra
                        ,[spectrum], axis=0)

            self._qsoLabels = np.ones(self._qsoSpectra.shape[0],dtype=np.int8)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")


    def _loadOthData(self):
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
                        loglam = np.pad(loglam, (0, 5000 - len(loglam)), mode='constant'
                        , constant_values=0)

                        flux = np.pad(flux, (0,5000 - len(flux)), mode='constant'
                        , constant_values=0)

                        spectrum = np.stack([loglam,flux])
                        self._othSpectra = np.append(self._othSpectra, 
                        [spectrum], axis=0)

            self._othLabels = np.zeros(self._othSpectra.shape[0],dtype=np.int8)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")

    def _combineDataLabels(self):
        """
            This combines both the arrays of labels into one array, to be fed 
            into the neural network.
        """

        self.labels = np.concatenate((self._qsoLabels,self._othLabels),axis=0)

    def _combineDataSpectra(self):
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
            self._loadQSOData()
            self._loadOthData()
            self._combineDataSpectra()
            return self.spectra

    @staticmethod
    def getLabels(self):
        """
            A simple method to retrieve the labels
        """
        if self.labels:
            return self.labels
        else:
            self._loadQSOData()
            self._loadOthData()
            self._combineDataLabels()
            return self.labels


dataLoader = loadData()

class preProcessor:
    def __init__(self,spectra=dataLoader.getSpectra):
        self._spectra = spectra
        self.outSpectra = None
        self._normalizedSpectra = None

    def _normalize(self):
        """
            Just, well normalizes the data.
        """

        normalizeValue = np.linalg.norm(self._spectra, 'fro')
        self._normalizedSpectra = self._spectra / normalizeValue
    
    def _randomShifts(self):
        """
            Add some very small random shifts to the spectra,to simulate noise.
        """

        shiftValues = np.array([-0.0001,-0.00009,-0.00008,-0.00007,-0.00006
        ,-0.00005,-0.00004,-0.00003,-0.00002,-0.00001
        ,0.00001, 0.00002,0.00003,0.00004,
        0.00005,0.00006,0.00007,0.00008
        ,0.00009,0.0001]
        ,dtype=np.longdouble) 
        self.outSpectra = np.copy(self._normalizedSpectra)
        numShifts = 24000 #just a random number
        for _ in range(numShifts):
            random2dSubarray = np.random.randint(0,self._normalizedSpectra.shape[0])
            random1dSubarray = np.random.randint(0,self._normalizedSpectra.shape[1])
            randomValue = np.random.randint(0,self._normalizedSpectra.shape[2])
            shiftValue = np.random.choice(shiftValues)
            self.outSpectra[random2dSubarray][random1dSubarray][randomValue] += shiftValue

        def getSpectra(self):
            """
                Just returns the preprocessed array
            """

            if self.outSpectra:
                return self.outSpectra
            else:
                self._normalize()
                self._randomShifts()
                return self.outSpectra




    
