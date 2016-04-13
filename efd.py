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
NUMBER_OF_HARMONICS = 20
SCALE = (9.8, 32.5) #TODO
SCALE_NUM = 20
#SCALE_NUM = None
IGNORE_LIST = []
OUTPUT_FILE = "log.txt"
COLOR = "b"
TRESCHOLD_VALUE = 140
COLUMNS = 4
ROWS = 5
RESIZE = None
KER1 = 15 #5

CUTING = False
Xi = 100
Yi = 200
Xe = 3900
Ye = 2600



def createLogFile( logFile, efds, cntAreas, referencePoints ):
    f = open( logFile, "w" )
    ii = 0
    for efd in efds:
        cntA = cntAreas[ii]
        
        f.write("label: "+str(ii)+"\r\n")
        f.write("area (mm2): "+str(cntA)+"\r\n")
        f.write("coordinate: "+str( referencePoints[ii] ) + "\r\n" )
        f.write("efd: \r\n")
        for item in efd:
            #f.write( "{0}".format(item) )
            f.write( str(item) )
            f.write("\r\n")
            
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
    scaleCnt = None
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
    
    ii = 0
    for imF in imList:
        if imF.split(".")[-1] in ["jpg", "JPG", "tiff"]:
            print ii
            ii += 1
            print imF
            efdMain( imF, n, directory )


def efdMain( imageFile, n = 6, directory = None ):
    if directory:
        img = cv2.imread( directory + imageFile, 1 )
    else:
        img = cv2.imread( imageFile, 1 )
        cv2.imwrite("logs/img1.png", img)
    
    scale = SCALE
    scaleNum = SCALE_NUM
    ignoreList = IGNORE_LIST
    color = COLOR
    tresh = TRESCHOLD_VALUE
    columns = COLUMNS
    rows = ROWS
    cuting = CUTING
    ker1 = KER1
    
    if cuting == True:
        img = cutImage(img, Xi, Yi, Xe, Ye, imShow = False )
        cv2.imwrite("logs/img2.png", img)
    
    gray = getGrayImg( img, color )
    cv2.imwrite("logs/img3.png", gray)
    binaryImg = getThreshold( gray, tresh )
    cv2.imwrite("logs/img4.png", binaryImg)
    binaryImg = openingClosing( binaryImg, ker1, ker2 = None )
    cv2.imwrite("logs/img5.png", binaryImg)
    
    contoursList, hierarchy= cv2.findContours( binaryImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contoursList, referencePoints = sortContours2(contoursList)
    
    contoursList, scaleCnt = getCleanList( contoursList, scaleNum, ignoreList ) #Cleanig
    referencePoints2, scalePoint = getCleanList( referencePoints, scaleNum, ignoreList )
    cv2.drawContours(img, contoursList, -1, (0,255,0), 1)
    #cv2.drawContours(gray, contoursList, -1, (255,255,255), 5) #for tae2016 only
    numCntF = rows * columns
    if len(contoursList) == 1:
        print "!!!!!!!!!!----- BLACK IMAGE -----!!!!!!!!!!"
        biLog = open( directory+"logs/black_images_log.txt", "a" )
        biLog.write(imageFile+"\r\n")
        biLog.close()
        
        return None
    
    assert len(contoursList) == numCntF, "len(contoursList) = %d" % len(contoursList)
    
    contoursList, referencePointsxy = sortContoursXY(contoursList, referencePoints2, columns, rows, ) #getting xy sorting
    contoursAreas = getContoursArea( contoursList )
    
    
    
    efds, K, T = elliptic_fourier_descriptors2( contoursList, n )
    
    #print "efds \n {0}".format(efds)
    
    if scaleNum is not None:
        scaleAreaP = cv2.contourArea( scaleCnt )
        scaleArea = scale[0] * scale[1] #Unit mm^2
        pixelSize = scaleArea / scaleAreaP
        cntAreas = getCntAreas( contoursAreas, pixelSize )
    else:
        cntAreas = contoursAreas
    
    #logFile = OUTPUT_FILE
    if directory:
        logFile = directory+"logs/"+imageFile.split(".")[0]+"_log.txt"
        createLogFile( logFile, efds, cntAreas, referencePointsxy )
        
        outFileNameLab = directory+"logs/"+imageFile.split(".")[0]+"_lab.png"
        #print outFileNameLab    
        resize = RESIZE
        writeLabelsInImg( img, referencePointsxy, outFileNameLab, referencePoints, resize )
        
        #outFileNameBi = directory+"logs/"+imageFile.split(".")[0]+"_bi.png"
        #cv2.imwrite( outFileNameBi, binaryImg )
        
    else:
        logFile = imageFile.split(".")[0]+"_log.txt"
        createLogFile( logFile, efds, cntAreas, referencePointsxy )
        
        outFileNameLab = imageFile.split(".")[0]+"_lab.png"
        #print outFileNameLab    
        resize = RESIZE
        writeLabelsInImg( img, referencePointsxy, outFileNameLab, referencePoints, resize )
        #writeLabelsInImg( gray, referencePointsxy, outFileNameLab, None, resize ) #for tae2016 only
        
        outFileNameBi = imageFile.split(".")[0]+"_bi.png"
        cv2.imwrite( outFileNameBi, binaryImg )


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
