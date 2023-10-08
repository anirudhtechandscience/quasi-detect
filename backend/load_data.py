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
        
    def loadQSOData(self):
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
                        spectrum = np.stack([loglam,flux])#stacks the two flux and loglam vectors into a matrix
                        self.qsoSpectra = np.append(self.qsoSpectra, [spectrum], axis=0)
        except FileNotFoundError:
            print("FileNotFoundError, please check if the path is valid")
        except Exception as e:
            print(f"A error occured : {e}")

ld = loadData()
ld.loadQSOData()