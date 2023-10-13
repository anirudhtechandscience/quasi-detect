from astropy.io import fits as fs 
import numpy as np
import os


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
            if fileName.endswith('.fits'):
                filePath= os.path.join(path, fileName)
                try:
                    self.spectra = np.append(spectra, self._getSpectrum(filePath))
                except FileNotFoundError:
                    print("FileNotFoundError")
                except Exception as e:
                    print(f"A error occured reading {filePath}: {e}")

    def _getSpectrum(self, filePath):
        """ returns the spectrum from a .fits file """

        with fs.open(filePath) as spec:
            binHDU = spec[1].data
            loglam = np.array(binHDU['loglam'], dtype=np.longdouble)
            flux = np.array(binHDU['flux'], dtype=np.longdouble)

        #TODO: Implement a dynamic maximum pad length system
        loglam = np.pad( loglam, (0, 5000 - len(loglam)), mode='constant', constant_values=0 )
        flux = np.pad( flux, (0,5000 - len(flux)), mode='constant', constant_values=0 )

        spectrum = np.stack( [loglam, flux] )        
        return spectrum
        


    def _loadAllData(self):
        """
            This loads the relevant columns from the .fits files , which is basically 
            the spectrum, and then makes a array with a 0 and 1 for every spectrum, 
            as the labels(Initially it will just tell quasars apart from other objects 
            to reduce complexity, later it will detect all kinds of objects). 
        """ 
        try:
            print("Loading other data")
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
                        print(self._othSpectra.shape)
                    
            print("Loading quasar data")

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
                        print(self._qsoSpectra.shape)

            self._othLabels = np.zeros(self._othSpectra.shape[0],dtype=np.int8)
            self._qsoLabels = np.ones(self._qsoSpectra.shape[0],dtype=np.int8) 
            self.spectra = np.concatenate((self._qsoSpectra,self._othSpectra),axis=0)
            self.labels = np.concatenate((self._qsoLabels,self._othLabels),axis=0)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")

    def getData(self):
        """
            A simple method to retrieve the data
        """
        if self.labels:
            return self.labels
        else:
            self._loadAllData()
            return (self.spectra,self.labels)


class preProcessor:
    def __init__(self,spectra):
        self._spectra = spectra[0]
        self.labels = spectra[1]
        self.outSpectra = None
        self._normalizedSpectra = None

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
            Add some very small random shifts to the spectra,to simulate noise.
        """

        shiftValues = np.array([-0.0001,-0.00009,-0.00008,-0.00007,-0.00006
        ,-0.00005,-0.00004,-0.00003,-0.00002,-0.00001
        ,0.00001, 0.00002,0.00003,0.00004,
        0.00005,0.00006,0.00007,0.00008
        ,0.00009,0.0001]
        ,dtype=np.longdouble) 
        self.outSpectra = np.copy(self._normalizedSpectra)
        numShifts = self._normalizedSpectra.shape[0] * 0.75
        numSubShift = 450 #every sub array will have this amount of values shifted
        self.outSpectra = np.copy(self._normalizedSpectra)
        for _ in range(numShifts):
            random2dSubarray = np.random.randint(0,self._normalizedSpectra.shape[0]) 
            twoDSubarray = np.copy(self._normalizedSpectra[random2dSubarray])
            for ___ in range(numSubShift):
                random1dSubarray = np.random.randint(0,self._normalizedSpectra.shape[1])
                randomValue = np.random.randint(0,self._normalizedSpectra.shape[2])
                shiftValue = np.random.choice(shiftValues)
                twoDSubarray[random1dSubarray][randomValue] += shiftValue
            self.outSpectra = np.append(self.outSpectra,[twoDSubarray],axis=0)

        def getData(self):
            """
                Just returns the preprocessed array
            """

            if self.outSpectra:
                return self.outSpectra
            else:
                self._normalize()
                self._randomShifts()
                return (self.outSpectra,self.labels)
