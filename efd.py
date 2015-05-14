#!/usr/bin/python
"""
  Elliptic fourier descriptors
    usage:
         python efd.py <switch> <Img|directory>
            switch: f - file, d - directory
"""
import cv2
#import cv2.cv as cv
import numpy as np
import os
import sys

from image import *
from elliptic_fourier_descriptors import *

#Constants
NUMBER_OF_HARMONICS = 10
SCALE = (20, 50) #TODO
SCALE_NUM = 100
IGNORE_LIST = [ 101, 102 ]
OUTPUT_FILE = "log.txt"
COLOR = "g"
TRESCHOLD_VALUE = 120
COLUMNS = 10
ROWS = 10


def createLogFile( logFile, efds, cntAreas ):
    f = open( logFile, "w" )
    ii = 0
    for efd in efds:
        cntA = cntAreas[ii]
        
        f.write("label: "+str(ii)+"\r\n")
        f.write("area (mm2): "+str(cntA)+"\r\n")
        f.write( "{0}".format(efd) )
        f.write("\r\n")
        
        ii += 1
    
    f.close()
    #TODO


def simplePicture( efds, T, K ):
    rec = reconstruct(efds[1,:],T[1],K[1])
    #scale to a fixed size for display purposes
    rec[:,0] = rec[:,0] - np.min(rec[:,0])
    rec[:,1] = rec[:,1] - np.min(rec[:,1])
    max_rec = np.max(rec[:,0])
    rec = rec / max_rec * 100
    
    max_tot = np.max(rec)
    #print max_tot
    imgN = np.zeros((max_tot+10, max_tot+10))
    
    for i in range(len(rec)):
        imgN[int(rec[i,0]), int(rec[i,1])] = 255
    
    cv2.imwrite("TMP/rec.png", imgN)


def getCleanList( oldList, scaleNum, ignoreList ):
    cleanList = []
    ii = 0
    for item in oldList:
        if ii == scaleNum:
            scaleCnt = item
            ii += 1
            continue
            
        elif ii in ignoreList:
            ii += 1
            continue
        
        cleanList.append( item )
        ii += 1
    
    return cleanList, scaleCnt


def getCntAreas( contoursAreas, pixelSize ):
    cntAreas = []
    for cntA in contoursAreas:
        cntAreas.append( cntA * pixelSize )
    
    return cntAreas


def efdFromDir( directory ):
    n = NUMBER_OF_HARMONICS
    imList = os.listdir( directory )
    print imList
    
    for imF in imList:
        print imF
        efdMain( imF, n, directory )


def efdMain( imageFile, n = 6, directory = None ):
    if directory:
        img = cv2.imread( directory + imageFile, 1 )
    else:
        img = cv2.imread( imageFile, 1 )
    
    scale = SCALE
    scaleNum = SCALE_NUM
    ignoreList = IGNORE_LIST
    color = COLOR
    tresh = TRESCHOLD_VALUE
    columns = COLUMNS
    rows = ROWS
     
    gray = getGrayImg( img, color )
    binaryImg = getThreshold( gray, tresh )
    binaryImg = openingClosing( binaryImg, ker1 = 5, ker2 = None )
    
    contoursList, hierarchy= cv2.findContours( binaryImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contoursList, referencePoints = sortContours2(contoursList)
    
    contoursList, scaleCnt = getCleanList( contoursList, scaleNum, ignoreList ) #Cleanig
    numCntF = rows * columns
    assert len(contoursList) == numCntF, "len(contoursList) = %d" % len(contoursList)
    
    contoursList, referencePoints = sortContoursXY(contoursList, referencePoints, columns = 10, rows = 10, ) #getting xy sorting
    contoursAreas = getContoursArea( contoursList )
    
    
    
    efds, K, T = elliptic_fourier_descriptors2( contoursList, n )
    
    print "efds \n {0}".format(efds)
    
    scaleAreaP = cv2.contourArea( scaleCnt )
    scaleArea = scale[0] * scale[1] #Unit mm^2
    pixelSize = scaleArea / scaleAreaP
    
    cntAreas = getCntAreas( contoursAreas, pixelSize )
    
    #logFile = OUTPUT_FILE
    logFile = "log_"+imageFile.split(".")[0]+".txt"
    createLogFile( "logs/"+logFile, efds, cntAreas )
    
    outFileNameLab = imageFile.split(".")[0]+"_lab.png"
    print outFileNameLab    
    writeLabelsInImg( img, referencePoints, "logs/"+outFileNameLab, resize = 0.3 )
    
    outFileNameBi = imageFile.split(".")[0]+"_bi.png"
    cv2.imwrite( "logs/"+outFileNameBi, binaryImg )


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
    
    n = NUMBER_OF_HARMONICS
    imageFile = sys.argv[2] #directory
    switch = sys.argv[1]
    if switch == "f":
        efdMain( imageFile, n)
    elif switch == "d":
        efdFromDir( imageFile )
