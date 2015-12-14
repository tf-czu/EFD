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

CUTING = True
Xi = 800
Yi = 350
Xe = 3400
Ye = 2700


def cutImage(img, xi, yi, xe, ye, imShow = False ):
    img = img[yi:ye, xi:xe]
    if imShow == True:
        showImg( img )
    return img


def writeLabelsInImg( img, referencePoints1, outFileName, referencePoints2 = None, resize = None ):
    num1 = 0
    color1 = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for point in referencePoints1:
        point = tuple(point)
        cv2.putText(img, str(num1),point, font, 3,color1,2 )
        num1 += 1
    
    if referencePoints2:
        offset = np.array([150, 0])
        num2 = 0
        color2 = (255, 0, 0)
        for point2 in referencePoints2:
            #print point2
            point2 = point2 + offset
            #print"point", point2
            point2 = tuple(point2)
            cv2.putText(img, str(num2),point2, font, 2,color2,2 )
            num2 += 1
            
    if resize:
        img = cv2.resize(img, None, fx = resize, fy = resize, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite( outFileName, img )
    return img


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


def openingClosing( binaryImg, ker1 = 10, ker2 = 5 ):
    newBinaryImg = None
    kernel = np.ones( ( ker1, ker1 ), np.uint8 )
    newBinaryImg = cv2.morphologyEx( binaryImg, cv2.MORPH_OPEN, kernel)
    
    if ker2 != None:
        kernel = np.ones( ( ker2, ker2 ), np.uint8 )
        newBinaryImg = cv2.morphologyEx( newBinaryImg, cv2.MORPH_CLOSE, kernel)
        
    return newBinaryImg


def imageMain( imageFile, tresh, color, cuting = True ):
    img = cv2.imread( imageFile, 1 )
    if cuting == True:
        img = cutImage(img, Xi, Yi, Xe, Ye, imShow = False )
        cv2.imwrite( "cutedImg.png", img )
        
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
        img = writeLabelsInImg( img, referencePoints, outFileNameLab )
        
        #cntNum = 100
        #cv2.drawContours(img, contoursList, cntNum, (0,255,0), 3)
        #cv2.imwrite( "imgLabCNT.png", img )


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
        
    cuting = CUTING
    imageFile = sys.argv[1]
    imageMain( imageFile, tresh, color, cuting )
