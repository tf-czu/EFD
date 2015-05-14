'''
Test code
'''

import cv2
import numpy as np
import os

from elliptic_fourier_descriptors import *
from contours import *


testShape = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
              [ 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [ 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
              [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
              [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],], np.uint8)*255


cv2.imwrite("TMP/shape.png", test_shape)
print test_shape

N = 4
shape = (10, 10)

#efds is a 3D array, 1st dim indexes different segments in the image,
#2nd dim indexes descriptors order, while 3d dimension is to access 
#the 4 different coefficients for each efd order
contoursList, hierarchy= cv2.findContours( test_shape1.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
contoursList, referencePoints = sortContours2(contoursList)
contoursList, referencePoints = sortContoursXY(contoursList, referencePoints, columns = 2, rows = 3, ) #getting xy sorting
contoursAreas = getContoursArea( contoursList )
efds, K, T = elliptic_fourier_descriptors2( contoursList, N)
print "K", K
print "T", T

print "shape \n {0}".format(efds)


#reconstruct the first shape

#access the first (and only) segment
rec = reconstruct(efds[0,:],T[0],K[0])
#scale to a fixed size for display purposes
rec[:,0] = rec[:,0] - np.min(rec[:,0])
rec[:,1] = rec[:,1] - np.min(rec[:,1])
max_rec = np.max(rec[:,0])
rec = rec / max_rec * 100


max_tot = np.max(rec)

img = np.zeros((max_tot+10,max_tot+10))

for i in range(len(rec)):
    img[int(rec[i,0]), int(rec[i,1])] = 255

cv2.imwrite("TMP/rec.png", img)

