"""
    This is a simple tool for drawing ellipses
        Usage:
            python drawing_ellipses.py abcd <number a> <number b> <number c> <number d>
            python drawung_ellipses.py f <file name> <order>
"""

import sys
import cv2
import numpy as np

from image import *
from elliptic_fourier_descriptors import *


#Create empty image
EM_IMAGE = np.ones((1000,1000, 3), np.uint8)*255


def scaleIm(xy, Xc = 500, Yc = 500, scale = 250 ):
    xy = xy*scale
    xy[:,0] = xy[:,0] + Xc
    xy[:,1] = xy[:,1] + Yc
    
    return xy


def xy2image( xyList ):
    img = EM_IMAGE
    color = (255, 0, 0)
    for ii in xrange( len( xyList ) - 1 ):
        pt1 = ( int( xyList[ ii ][0] ), int( xyList[ ii ][1] ) )
        pt2 = ( int( xyList[ ii + 1 ][0] ), int( xyList[ ii + 1 ][1] ) )
        cv2.line(img, pt1, pt2, color, thickness = 2 )
    return img
    

def giveEfdXY( t, efdL, Xc = 500, Yc = 500, scale = 250 ):
    xyList = []
    for tt in t:
        x = 0
        y = 0
        for efd in efdL:
            a = efd[0]
            b = efd[1]
            c = efd[2]
            d = efd[3]
            print a, b, c, d
            xn = a * np.cos(tt) + b * np.sin(tt)
#            print "xn", xn
            yn = c * np.cos(tt) + d * np.sin(tt)
#            print "yn", yn
            x = x + xn
            y = y + yn
        x = x * scale + Xc
        y = y * scale + Yc
        x = int( np.round(x))
        y = int( np.round(y))
        xyList.append( [x, y] )
    return xyList
    

def giveXY( t, a = 1, b = 1, c = 1, d = 1 ):
    xyList = []
    for tt in t:
        x = a*np.cos(tt) + b*np.sin(tt)
        y = c*np.sin(tt) + d*np.sin(tt)
        xyList.append([x, y])
    
    return xyList


def drawSimpleEllipse( a, b, c, d, tn = 200):
    t = np.linspace( 0, 2*np.pi, tn )
    xyList = giveXY( t, a, b, c, d )
    xy = np.array(xyList)
    xy = scaleIm(xy)
    
    img = xy2image( xy )
        
    showImg( img )
    writeImg( img, "testIm.png" )


def shapeFromFile( fileName, order, N ):
    #tn is number of points (T?)
    #t = np.linspace( 0, 2*np.pi, tn )
    f = open( fileName, "r" )
    nn = 0
    ordN = 0
    efdL = [[0, 0, 0, 0]]
    for line in f:
        if line[0] == "#":
            continue
        if line[0] != " ":
            ordN += 1
            continue
        if ordN > order:
            break

        if ordN == order:
            nn += 1
            lineL = line.split()
            lineF = map( float, lineL )
            efdL.append( lineF )
            if nn == N:
                break
    print efdL
    efd = np.array(efdL)
    #xyList = giveEfdXY( t, efds )
    rec = reconstruct(efd, T = 2810.5, K = 40)
    
    rec = scaleIm(rec)
    print "rec", rec
    img = xy2image( rec )
    showImg( img )
    writeImg( img, "testIm.png" )


def shapeFromFile2( fileName, order, N ):
    f = open( fileName, "r" )
    efdL = []
    nn = 0
    #ordN = 0
    read = False
    label = False
    for line in f:
        line = line.split(":")
        if (line[0] == "label") and (float(line[1]) == order):
            label = True
            print line
            print "label = True"
            continue
        
        if (label == True ) and (line[0] == "efd"):
            read = True
            continue
            
        if (read == True) and (label == True):
            lineList = line[0][1:-3].split()
            #print lineList
            abcd = map( float, lineList )
            print abcd
            efdL.append(abcd)
            nn += 1
            if nn == N:
                break
                
    print efdL
    efd = np.array(efdL)
    #xyList = giveEfdXY( t, efds )
    rec = reconstruct(efd, T = 2810.5, K = 40)
    
    rec = scaleIm(rec)
    print "rec", rec
    img = xy2image( rec )
    showImg( img )
    writeImg( img, "testIm.png" )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
        
    N = None
    if sys.argv[1] == "abcd":
        a = float(sys.argv[2])
        b = float(sys.argv[3])
        c = float(sys.argv[4])
        d = float(sys.argv[5])
        drawSimpleEllipse( a, b, c, d )
    
    #data from file
    if sys.argv[1] == "f":
        fileName = sys.argv[2]
        order = int( sys.argv[3] )
        if len( sys.argv ) > 4:
            N = int( sys.argv[4] )
        shapeFromFile( fileName, order, N )
        
    if sys.argv[1] == "f2":
        fileName = sys.argv[2]
        order = int( sys.argv[3] )
        if len( sys.argv ) > 4:
            N = int( sys.argv[4] )
            
        shapeFromFile2( fileName, order, N )
