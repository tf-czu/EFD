#!/usr/bin/python
"""
  Simple tools for images
    usage:
         python image.py <input img> <color> <treshold value>
"""

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

from contours import *


def writeLabelsInImg( img, referencePoints, outFileName ):
    num = 0
    color = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for point in referencePoints:
        point = tuple(point)
        cv2.putText(img, str(num),point, font, 4,color,2 )
        num += 1
    
    cv2.imwrite( outFileName, img )


def writeImg( img, fileName ):
    cv2.imwrite( fileName, img )

def showImg( img ):
    cv2.imshow( 'image', img )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getHist( grayImg ):
#    showImg( grayImg )
    plt.hist( grayImg.ravel(), 256,[0,256])
    plt.show()


def getGrayImg( img, color = "g" ):
    gray = None
    b,g,r = cv2.split( img )
    if color == "g":
        gray = g
    elif color == "b":
        gray = b
    elif color == "r":
        gray = r
    else:
        print "color is not defined!"
    return gray


def getThreshold( gray, thrValue ):
    ret, binaryImg = cv2.threshold( gray, thrValue, 255,cv2.THRESH_BINARY_INV)
    return binaryImg


def openingClosing( binaryImg, ker1 = 5, ker2 = 2 ):
    newBinaryImg = None
    kernel = np.ones( ( ker1, ker1 ), np.uint8 )
    newBinaryImg = cv2.morphologyEx( binaryImg, cv2.MORPH_OPEN, kernel)
    
    if ker2 != None:
        kernel = np.ones( ( ker2, ker2 ), np.uint8 )
        newBinaryImg = cv2.morphologyEx( newBinaryImg, cv2.MORPH_CLOSE, kernel)
        
    return newBinaryImg


def imageMain( imageFile, tresh, color ):
    img = cv2.imread( imageFile, 1 )
    gray = getGrayImg( img, color )
    
    grayImgName = imageFile.split(".")[0]+"_gray.png"
    cv2.imwrite( grayImgName, gray )
    
    getHist( gray )
    if tresh:
        binaryImg = getThreshold( gray, tresh )
        binaryImg2 = openingClosing( binaryImg, ker1 = 5, ker2 = None )
        
        newImgName = imageFile.split(".")[0]+"_test.png"
        newImgName2 = imageFile.split(".")[0]+"_test2.png"
        #newImgName = "newImg.png"
        cv2.imwrite( newImgName, binaryImg )
        cv2.imwrite( newImgName2, binaryImg2 )
        
        contoursList, hierarchy= cv2.findContours( binaryImg2.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contoursList, referencePoints = sortContours2(contoursList)
        print "Number of cnt:", len(contoursList)
        
        outFileNameLab = imageFile.split(".")[0]+"_lab.png"
        #print outFileNameLab
        writeLabelsInImg( img, referencePoints, outFileNameLab )


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
    
    color = "g"
    tresh = None
    if len(sys.argv) > 2:
        color = sys.argv[2]
    
    if len(sys.argv) > 3:
        tresh = float(sys.argv[3])
        
    imageFile = sys.argv[1]
    imageMain( imageFile, tresh, color )
