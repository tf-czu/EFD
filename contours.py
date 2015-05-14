"""
  Elliptic fourier descriptors
    usage:
         python efd.py <input Img>
"""

import sys
import numpy as np
import cv2


def getContoursArea( contours, scale = 1.0 ):
    contoursAreas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        contoursAreas.append( area * scale )
    
    return contoursAreas


def sortContours2( contours, direction = "x" ):#TODO
    contourPoints = np.zeros((len(contours),2), dtype = int)
    if direction == "x":
        a = 1
        b = 0
    elif direction == "y":
        a = 0
        b = 1
    
    counter = 0
    for cnt in contours:
        conResh = np.reshape(cnt,(-1,2))
        idx = np.lexsort( (conResh[:,a],conResh[:,b]) )
        sortedContours = conResh[idx,:]
        contourPoints[counter,:] = sortedContours[0,:] # The coordinate of reference point.
        counter = counter + 1
        
    sortedIdx = np.lexsort((contourPoints[:,a], contourPoints[:,b]))
    sortedContours = []
    referencePoints = []
    for idx in sortedIdx:
        sortedContours.append(contours[idx])
        referencePoints.append(contourPoints[idx])
        
    return sortedContours, referencePoints


def sortContoursXY( contours, referencePoints, columns = 10, rows = 10, direction = "xy" ): #ver 0
    contoursXYSorted = []
    refPointsList = []
    refPointsF = []
    sortedCntF = []
    
    #print referencePoints
    for ii in xrange(columns):
        index0 = ii * rows
        index1 = ( ii + 1 ) * rows
        #print index0, index1
        pList = np.array( referencePoints[ index0 : index1 ] )
        cList = contours[ index0 : index1 ]
        #print "pList, cList", pList, cList
        refPointsList.append(pList)
        contoursXYSorted.append(cList)
    #print refPointsList
    #print contoursXYSorted
    
    kk = 0
    for pList in refPointsList:
        cList = contoursXYSorted[kk]
        kk += 1
        #print pList
        idy = np.lexsort( (pList[:,0],pList[:,1]) )
        #print idy
        for y in idy:
            refPointsF.append( pList[y] )
            sortedCntF.append( cList[y] )
    
    #print sortedCntF
    return sortedCntF, refPointsF




def contoursMain( binaryImgFile ):
    pass
    

if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
    binaryImgFile = sys.argv[1]
    contoursMain( binaryImgFile )
