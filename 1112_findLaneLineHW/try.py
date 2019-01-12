import numpy as np

objp=np.zeros((6*9,3),np.float32)
print(np.mgrid[0:9, 0:6].T)
objp[:,:2] = np.mgrid[0:9, 0:6].T