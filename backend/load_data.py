from astropy.io import fits as fs 
import numpy as np
import os


#This class uses standard Python conventions, such as _ to indicate private objects
#Also, I use CamelCase across the entire codebase, both frontend and backend


class loadData:

    def __init__(self,qsoPath="/home/tux/Downloads/QSO",othPath="/home/tux/Downloads/oth"):
        self._dirPathQuasar = qsoPath
        self._dirPathOther = othPath
        self.qsoSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self.othSpectra = np.empty((0, 2, 5000), dtype=np.longdouble)
        self.othLabels = None
        self.qsoLabels = None
        
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
                        self.qsoSpectra = np.append(self.qsoSpectra, [spectrum], axis=0)
            self.qsoLabels = np.ones(self.qsoSpectra.shape[0])
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
                        self.othSpectra = np.append(self.othSpectra, [spectrum], axis=0)
            self.othLabels = np.zeros(self.othSpectra.shape[0])
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")

    
