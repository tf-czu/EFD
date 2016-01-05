"""
  Tools for analyse of logsfiles
    usage:
         python data.py <switch> <directory>
            switch: efd
"""

import sys
import os
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data import *


IGNORE_LIST = []
EFD_COEFFICIENTS = [7, 11, 12, 15, 19, 20]
#EFD_COEFFICIENTS = [7]
DATA_OUTPUT = [ "plots_1d", "plots_2d", "plots_3a", "plots_3d", "plots_4d", "plots_5a" ]
DEGREE = 3
FRAMES_PER_H = 30.0
SMOOTH_NUM_AREA = 10
SMOOTH_NUM_AREA_DETAIL = 4
DETAIL_AREA = 6 #time in hours
OFFSET = 120
LEN_AREA_G1 = 40
LEN_AREA_G2 = 20 #30
LEN_AREA_G3 = 10

LEN_EFD_G1 = 120
LEN_EFD_G2 = 60
LEN_EFD_G3 = 30

#Fit model 5
GRID1 = ( 0, 50.0, 25001 )
GRID2 = ( 0, 5.0, 2501 )
GRIDF = 401


createAreaPlots = False
createAreaPlotsGrad = False
createAreaDetail = False
areaAnalyse2 = True
createEfdsPlotsOrig = False
createEfdsPlotsGrad1 = False
createEfdsPlotsGrad2 = False
createEfdsPlotsPoly = False
createEfdsPlotsPoly2 = False
createEfdsPlotsPoly3 = False


#def model5( x, S0, a, k1, k2, d, a0, k0 = 0.001 ):
#    return S0 + a + a/( k1 - k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k1*( 1 - d )* np.exp(-k2*x ) ) + a0*( np.exp( k0*x ) -1 )
  
def model5( x, S0, a, k1, k2, d, a0 ):
    return S0 + a + a/( k1 - k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k1*( 1 - d )* np.exp(-k2*x ) ) + a0*x

def model5a( x, a, k1, k2 ):
    return a*( 1 + 1/( k1-k2 ) * ( np.exp(-k1*x )*k2 - k1*np.exp(-k2*x ) ) )
    
def model5ad( x, ad, k1, k2 ):
    return ad*k1/( k1-k2 ) * ( np.exp(-k2*x ) - np.exp(-k1*x ) )
    
#def model5a0( x, a0, k0 ):
#    return a0*( np.exp( k0*x ) -1 )
    
def model5a0( x, a0 ):
    return a0*x
    
def model5DI( x, a, k1, k2, d, a0 ):
    return -a*k1/( k1 -k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k2*( 1 - d )* np.exp(-k2*x ) ) + a0
    
def model5DII( x, a, k1, k2, d ):
    return -a*k1/( k1 -k2 ) * ( k1*( -k2 + k1*d )* np.exp(-k1*x ) + k2**2 *( 1 - d )* np.exp(-k2*x ) )



def divideMatrix( myArr ):
    arrA = myArr.copy()
    arrB = myArr.copy()
    for ii in xrange( myArr.shape[0] ):
        arrA[ ii, ii:] = np.nan
        arrB[ ii, :ii] = np.nan

    return arrA, arrB
    

def getLocalMin( myAr ):
    r, c = myAr.shape
    localMinList = []
    minValueList = []
    step = 0.1
    for rr in xrange(r):
        if rr <= 1 or rr >= ( r - 2 ):
            continue
            
        for cc in xrange(c):
            if cc <= 1 or cc >= ( c - 2 ):
                continue
                
            x00 = myAr[rr,cc]
            if np.isnan(x00):
                continue
            localGrup = myAr[ rr-2:r+2, cc-2:cc+2 ]
            if x00 == np.nanmin( localGrup ):
                localMinList.append( [rr,cc] )
                minValueList.append( x00 )
        
        counter = rr/float(r)
        if counter >= step:
            print "Local min: ", round(step * 100), "%"
            step += 0.1
        
    if len( localMinList ) > 0:
        absMin = localMinList[ np.argmin( minValueList ) ]
        return localMinList, tuple( absMin ), minValueList
    else:
        return None, None, None
    
    


def fitModel5( x, y, seedId, gridK1, gridK2, sigma = None, oneSide = False ):
    k1f = gridK1[0]
    k1l = gridK1[1]
    k2f = gridK2[0]
    k2l = gridK2[1]
    k1n = gridK1[2]
    k2n = gridK2[2]
    k1Ar = np.linspace( k1f, k1l, k1n )
    k2Ar = np.linspace( k2f, k2l, k2n )
    diffYYpNormAr = np.zeros( ( k1n, k2n ) )
    coeffAr = np.zeros( ( k1n, k2n, 6 ) )
    solutionLog = None
    
    ii = 0
    step = 0.1
    for k1 in k1Ar:
        jj = 0
        
        for k2 in k2Ar:
            a, d, a0, S0 = None, None, None, None
            diffYYpNorm = None
            if k2 == 0 or k1 == 0:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            if k1 == k2:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
            
            if oneSide == True and k1 < k2:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            partA = model5a( x, 1, k1, k2 )
            partAD = model5ad( x, 1, k1, k2 )
            #partA0 = model5a0( x, 1, k0 )
            partA0 = model5a0( x, 1 )
            partList = [ partA, partAD, partA0, np.ones( len(x) ) ]
            A = np.zeros( [4,4] )
            B = np.zeros((4,1))
            for rr in range(4):
                for cc in range(4):
                    A[rr, cc] =  np.sum( np.multiply( partList[rr], partList[cc] ) )
            for rr in range(4):
                B[rr, 0] = np.sum( np.multiply( y, partList[rr] ) )
            
            try:
                X = np.linalg.solve(A, B)
            except:
                if solutionLog is None:
                    slogName = "logs2/area/solutinLog"+str(seedId)+".txt"
                    solutionLog = open( slogName, "w")
                    solutionLog.write( "ii, jj, k1, k2\r\n" )
                    print "The solutinLog was created!"
                    
                solutionLog.write("%d, %d, %f, %f\r\n" % ( ii, jj, k1, k2 ) )
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            #print X
            a, ad, a0, S0 = X
            d = ad/a
            #coeffAr[ii,jj] = S0, a, k1, k2, d, a0, k0
            #yp = model5( x, S0, a, k1, k2, d, a0, k0 )
            coeffAr[ii,jj] = S0, a, k1, k2, d, a0
            yp = model5( x, S0, a, k1, k2, d, a0 )
            diff = y - yp
            if sigma is not None:
                diff = diff/sigma
            diffYYpNorm = np.linalg.norm( diff )
            #print diffYYpNorm
            diffYYpNormAr[ii,jj] = diffYYpNorm
            
            jj += 1
            #if ii > 20:
            #    sys.exit()
            
        ii += 1
        counter = ii/float(k1n)
        if counter >= step:
            print "Fit model: ", round(step * 100), "%"
            step += 0.1
    
    if solutionLog is not None:
        solutionLog.close()
    
    return diffYYpNormAr, coeffAr


def fitLineAndLine( x, y, startTime = 1.0, endTime = 24.0 ):
    fph = FRAMES_PER_H
    startDp = int( round(startTime*fph) )
    endPoint = int( round(endTime*fph) )
    for dp in xrange( startDp, endPoint + 1 ):
        # a1*x + b1; a2*x + b2
        # a1*xp + b1 = a2*xp + b2
        xp = x[dp]
        x1 = x[:dp]
        x2 = x[dp:]
        y1 = y[:dp]
        y2 = y[dp:]
        A = np.zeros( [4,4] )
        sx21 = np.sum( np.power( x1, 2 ) )
        sx11 = np.sum( x1 )
        sx22 = np.sum( np.power( x2, 2 ) )
        sx12 = np.sum( x2 )
        A[0,:] = [ xp, 1, -xp, -1 ]
        A[1,:] = [ sx21, sx11, 0, 0 ]
        A[2,:] = [ sx11, dp, 0, 0 ]
        A[3,:] = [ 0, 0, sx22, sx12 ]
        B = np.zeros((4,1))
        syx1 = np.sum( y1*x1 )
        sy1 = np.sum( y1 )
        syx2 = np.sum( y2*x2 )
        B[:,0] = [ 0, syx1, sy1, syx2 ]
        
        cp = np.linalg.norm(A)*np.linalg.norm( np.linalg.inv(A) )
        print "Cp: ", cp
        
        X = np.linalg.solve(A, B)
        print A
        print B
        print X
        a1, b1, a2, b2 = np.reshape(X,-1)
        print a1, b1, a2, b2
        
        yp = np.zeros(len(y))
        pl1 = np.poly1d([a1, b1])
        pl2 = np.poly1d([a2, b2])
        yp[:dp] = pl1( x[:dp] )
        yp[dp:] = pl2( x[dp:] )
        ymean = np.sum(y)/len(y)
        ssreg = np.sum((yp-ymean)**2)
        sstot = np.sum((y - ymean)**2)
        rSquared = ssreg/sstot
        
        normdiffYYp = np.linalg.norm(y-yp)
        
        plt.figure()
        plt.plot(x,y,"k")
        plt.plot(x[:dp],yp[:dp],"r")
        plt.plot(x[dp:],yp[dp:],"g")
        plt.text(20, 8500, str(normdiffYYp), fontsize=12)
        #plt.show()
        plt.savefig("logs2/area/tmp/area_Plot_%003d" %dp, dpi=100)
        plt.clf
        plt.close()
        #sys.exit()


def fitPolyAndLine( x, y, line1, dPoint1, endTime, ver = "ver1" ):
    #a2*x^4 + b2*x^3 + c2*x^2 + d2*x +e
    #a3*x + b3
    fph = FRAMES_PER_H
    coeffsList = []
    rSquaredList = []
    dPoitList = []
    startD2Point = dPoint1 + 10
    endpoint = int( endTime * fph )
    a1 = line1[0]
    if ver == "ver1":
        xp1 = x[dPoint1]
    elif ver == "ver2":
        xp1 = 0
    
    for d2p in xrange( startD2Point, endpoint + 1 ):
        yl3 = y[d2p:]
        xl3 = x[d2p:]
        a3, b3 = np.polyfit(xl3, yl3, 1)
        
        xp2 = x[d2p]
        xp = x[ dPoint1 : d2p ]
        yp = y[ dPoint1 : d2p ]
        
        if ver == "ver1":
            A = np.zeros( (5,5) )
            A[0,:] = [ xp1**4, xp1**3, xp1**2, xp1, 1 ]
            A[1,:] = [ 4*xp1**3, 3*xp1**2, 2*xp1, 1, 0 ]
            A[2,:] = [ xp2**4, xp2**3, xp2**2, xp2, 1 ]
            A[3,:] = [ 4*xp2**3, 3*xp2**2, 2*xp2, 1, 0 ]
            #for nn in range(5):
            #    A[4,nn] = np.sum( np.power( xp, 8 - nn ) )
            for nn in range(5):
                A[4,nn] = np.sum( np.power( xp, 4 - nn ) )
            A[4,:] = A[4,:]/1000.0
            B = np.zeros((5,1))
            #sumX4Y = np.sum( np.power( xp, 4 ) * yp )
            sumX4Y = np.sum( yp )
            B[:,0] = [ a1*xp1, a1, a3*xp2 + b3, a3, sumX4Y/1000.0 ]
            
            #A = np.zeros( (4,4) )
            #A[0,:] = [ xp1**3, xp1**2, xp1, 1 ]
            #A[1,:] = [ 3*xp1**2, 2*xp1, 1, 0 ]
            #A[2,:] = [ xp2**3, xp2**2, xp2, 1 ]
            #A[3,:] = [ 3*xp2**2, 2*xp2, 1, 0 ]
            #B = np.zeros((4,1))
            #B[:,0] = [ a1*xp1, a1, a3*xp2 + b3, a3 ]
            
        elif ver == "ver2":
            A = np.zeros( (3,3) )
            A[0,:] = [ xp2**4, xp2**3, xp2**2 ]
            A[1,:] = [ 4*xp2**3, 3*xp2**2, 2*xp2 ]
            for nn in range(3):
                A[2,nn] = np.sum( np.power( xp, 6 - nn ) )
        B = np.zeros((3,1))
        sumX2Y = np.sum( np.power( xp, 2 ) * yp )
        B[:,0] = [ a3*xp2 + b3, a3, sumX2Y ]
        
        cp = np.linalg.norm(A)*np.linalg.norm( np.linalg.inv(A) )
        print "Cp: ", cp
        
        X = np.linalg.solve(A, B)
        print A
        print B
        print X
        coeffsP = np.reshape(X,-1)
        print coeffsP
        #sys.exit()
        plt.figure()
        if ver == "ver1":
            pl1 = np.poly1d( line1 )
            ypl1 = pl1( x[:dPoint1] )
            plt.plot(x[:dPoint1],ypl1,"b")
            pp2 = np.poly1d( coeffsP )
        elif ver == "ver2":
            a2, b2, c2 = coeffsP
            pp2 = np.poly1d( [a2, b2, c2, 0, 0] )
        
        pl3 = np.poly1d( (a3,b3) )
        ypp2 = pp2(xp)
        ypl3 = pl3( x[d2p:] )
        #ybar = np.sum(y)/len(y)
        #ssreg = np.sum((yp-ybar)**2)
        #sstot = np.sum((y - ybar)**2)
        #rSquared = ssreg/sstot
        #rSquaredList.append(rSquared)
        plt.plot(x,y,"k")
        plt.plot(xp,ypp2,"r")
        plt.plot(x[d2p:],ypl3,"g")
        #plt.show()
        plt.savefig("logs2/area/tmp/area_Plot_%003d" %d2p, dpi=100)
        plt.clf
        plt.close()
    
    sys.exit()
    #return coeffsPoly, coeffsLine2, dPoint2, rSquaredPL



def myFunctionFit( x, y ):
    #y = a*x^3 + b*x^2
    ATA = np.zeros((2,2))
    K = np.zeros((2,1))
    ATA[0,0] = np.sum( np.power(x, 6) )
    ATA[0,1] = np.sum( np.power(x, 5) )
    ATA[1,0] = np.sum( np.power(x, 5) )
    ATA[1,1] = np.sum( np.power(x, 4) )
    yx3 = np.multiply( y, np.power(x, 3) )
    yx2 = np.multiply( y, np.power(x, 2) )
    K[0,0] = np.sum( yx3 )
    K[1,0] = np.sum( yx2 )
    ab = np.linalg.solve(ATA, K)
    
    return ab


def searchLine( x, y, minTime = 0.5, constantB = None, timeLimit = 24.0 ):
    searchContinue = True
    widthR = 10
    fph = FRAMES_PER_H
    inp = int( round(minTime * fph) ) #initial number of point
    rSquaredList = []
    coeffsList = []
    ii = 0
    while searchContinue == True:
        inputX = x[ : inp + ii ]
        inputY = y[ : inp + ii ]
        if constantB == None:
            yp, rSquared, coeffs = dataPolyFit( inputX, inputY, 1, returnCoeffs = True )
            a, b = coeffs
            
        else:
            #constantB = 0
            b = 0
            sumXsq = np.sum( np.power( inputX, 2 ) )
            sumXY = np.sum( np.multiply( inputX, inputY ) )
            a = sumXY/sumXsq
            
            yp = np.zeros(len(inputY))
            p = np.poly1d( (a,b) )
            yp = p(inputX)
            yMean = np.mean(inputY)
            ssreg = np.sum((yp-yMean)**2)
            sstot = np.sum((inputY - yMean)**2)
            rSquared = ssreg/sstot
            
        rSquaredList.append(rSquared)
        coeffsList.append([a, b])
        
        if ii > timeLimit * fph:
            searchContinue = False
            print "ii > timeLimit * fph" #TODO
        
        if ii >= widthR - 1:
            ar, br = np.polyfit( np.arange(widthR), rSquaredList[-widthR:] , 1)
            #print ar
            if ar < 0:
                coeffsFinal = coeffsList[ -widthR ]
                rSquaredF = rSquaredList[ -widthR ]
                point = inp + ii - widthR
                endTime = float( point )/fph
                searchContinue = False
            
        ii += 1
        
    #print rSquaredList
    return coeffsFinal, endTime, rSquaredF, point


def findY(x, y, xi):
    xDiff = abs(x - xi)
    idx = np.argmin(xDiff)
    
    return y[idx]


def getExtremes( x, y, xd, yd ):
    result = []
    for ii in xrange( len(yd) - 1 ):
        y0 = yd[ii]
        y1 = yd[ii+1]
        if (y0 * y1) > 0:
            continue
            
        if y0 > y1: #maximum
            xi = ( xd[ii] + xd[ii] ) /2.0
            yi = findY(x, y, xi)
            result.append(["max", xi, yi])
            
        else: #minimum
            xi = ( xd[ii] + xd[ii] ) /2.0
            yi = findY(x, y, xi)
            result.append(["min", xi, yi])
        
    return result


def extremsPoly3Type( x, coeffs ):
    a, b, c, d = coeffs
    yd2 = 6*a*x + 2*b
    if yd2 > 0:
        extremTyp = "min"
    elif yd2 < 0:
        extremTyp = "max"
    else:
        extremTyp = None
        
    return extremTyp


def extremsFromPoly3( coeffs ):
    a, b, c, d = coeffs
    #first derivative: y' = 3ax^2 + 2bx + c
    #formula: x1,2 = ( -2b +- sqrt( 4b^2 - 12ac ) ) / 6a
    #second derivative: y" = 6ax + 2b
    #formula: x = -b/(3a)
    
    if a == 0:
        x1 = -c/( 2*b )
        x2 = None
        xInf = None
        
    else:
        D = 4*b**2 - 12*a*c
        
        if D > 0: 
            x1 = ( -2*b + np.sqrt( D ) ) / (6*a)
            x2 = ( -2*b - np.sqrt( D ) ) / (6*a)
            
        elif D == 0:
            x1 = ( -2*b ) / (6*a)
            x2 = None
            
        else:
            x1 = None
            x2 = None
            
        xInf = -b/( 3*a )
    
    if x1:
        exTyp1 = extremsPoly3Type( x1, coeffs )
        if exTyp1:
            ex1F = [exTyp1, x1]
        else:
            ex1F = None
    else:
        ex1F = None
            
    if x2:
        exTyp2 = extremsPoly3Type( x2, coeffs )
        if exTyp2:
            ex2F = [exTyp2, x2]
        else:
            ex2F = None
    else:
        ex2F = None
    
    return ex1F, ex2F, xInf


def dataPolyFit( x, y, degree, returnCoeffs = False ):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yp = p(x)
    #calculaton of  the r-squared (coefficient of determination)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yp-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    rSquared = ssreg/sstot
    
    if returnCoeffs == True:
        return yp, rSquared, coeffs
    else:
        return yp, rSquared


def myGradient( x, y, length1, length2 = None, length3 = None ):
    length1 = int(length1)
    n = len(y)
    
    yOut1 = np.zeros( n-length1 )
    xOut1 = np.zeros( n-length1 )
    yOut2 = None
    xOut2 = None
    yOut3 = None
    xOut3 = None
    for ii in range( n-length1 ):
        item = y[ ii : ii + length1 ]
        xItem = x[ ii : ii + length1 ]
        a, b = np.polyfit( xItem, item, 1 )
        yOut1[ii] = a
        xOut1[ii] = ( x[ ii + length1/2 ] + x[ ii + length1/2 + 1 ] ) / 2.0
        
    if length2:
        length2 = int(length2)
        n2 = len(yOut1)
        yOut2 = np.zeros( n2-length2 )
        xOut2 = np.zeros( n2-length2 )
        for ii in range(n2-length2):
            item = yOut1[ ii : ii + length2 ]
            xItem = xOut1[ii : ii + length2]
            a, b = np.polyfit( xItem, item, 1 )
            yOut2[ii] = a
            xOut2[ii] = ( xOut1[ ii + length2/2 ] + xOut1[ ii + length2/2 + 1 ] ) / 2.0
        
    if length2 and length3:
        length3 = int(length3)
        n3 = len(yOut2)
        yOut3 = np.zeros( n3-length3 )
        xOut3 = np.zeros( n3-length3 )
        for ii in range(n3-length3):
            item = yOut2[ ii : ii + length3 ]
            xItem = xOut1[ii : ii + length3]
            a, b = np.polyfit( xItem, item, 1 )
            yOut3[ii] = a
            xOut3[ii] = ( xOut2[ ii + length3/2 ] + xOut2[ ii + length3/2 + 1 ] ) / 2.0
        
    return xOut1, yOut1, xOut2, yOut2, xOut3, yOut3
    


def getSmoothData( y, smoothNum ):
    window = np.ones( smoothNum ) / smoothNum
    yOut = np.zeros( len(y) )
    yCon = np.convolve( y, window, "valid")
    #yOut = np.convolve( y, window, "same")
    yOut[ smoothNum -1 : ] = yCon
    yOut[ 0:smoothNum-1 ] = yCon[0]
    return yOut
    

def getSmoothData2( x, y, smoothNum, outlier = True ):
    n = len(y) - smoothNum
    yOut = []
    xOut = x[ smoothNum/2 : n + smoothNum/2 ]
    
    for ii in xrange(n):
        itemY = y[ ii : smoothNum + ii ]
        itemX = x[ ii : smoothNum + ii ]
        if (outlier == True) and (itemX[0] >= 0):
            coeffs = np.polyfit( itemX, itemY, 1)
            p = np.poly1d(coeffs)
            itemYp = p(itemX)
            diff = itemYp - itemY
            stdDiff = np.std(diff)
            sumItemY = 0
            num = 0
            numOfOutlier = 0
            
            for jj in range( len(diff) ):
                if abs( diff[jj] ) < 1.0 * stdDiff: # sigma limit
                    sumItemY += itemY[jj]
                    num += 1
                else:
                    #print "outlier"
                    #print stdDiff, diff, jj
                    numOfOutlier += 1
                
            if numOfOutlier > 1:
                print "numOfOutlier", numOfOutlier
                
            if num == 0:
                print "diff", diff
                print "stdDiff", stdDiff
                sys.exit()
            print "num", num
            meanItemY = sumItemY/num
            
        else:
            meanItemY = np.mean(itemY)
                
        yOut.append(meanItemY)
        
    yOut = np.array(yOut)
    
    return xOut, yOut


def getIdNums( idNumsIn, ignorList ):
    idNumsOut = []
    offset = 0
    for item in idNumsIn:
        if item in ignorList:
            offset += 1
            
        idNumsOut.append(item + offset )
        
    return idNumsOut


def efdAnalyse1( directory ):
    subject = "efd"
    subject_area = "area"
    ignorList = IGNORE_LIST
    efdsCoeff = EFD_COEFFICIENTS
    dataOutPut = DATA_OUTPUT
    fph = FRAMES_PER_H
    detailA = DETAIL_AREA
    smoothNumA = SMOOTH_NUM_AREA
    smoothNumAD = SMOOTH_NUM_AREA_DETAIL
    lenAG1 = LEN_AREA_G1
    lenAG2 = LEN_AREA_G2
    lenAG3 = LEN_AREA_G3
    lenG1 = LEN_EFD_G1
    lenG2 = LEN_EFD_G2
    lenG3 = LEN_EFD_G3
    offset = OFFSET
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    areaAr = readData( fileList, directory, subject_area, ignorList )
    #np.savetxt("area_data", areaAr)
    #sys.exit()
    
    d1, d2, d3 = efds.shape
    print d1, d2, d3
    x = ( np.arange(d1) ) * 1/fph
    idNums = getIdNums( range(d2), ignorList )
    
    if createAreaPlotsGrad == True:
        log = open("logs2/area/extremes_area.txt", "w")
        for ii in xrange(d2):
            y = areaAr[:, ii]
            #ai, bi = np.polyfit( x[:15], y[:15], 1 )
            coeffsLine, dp1Time, rSquared, dPoint1 = searchLine( x, y, minTime = 0.5, timeLimit = 10.0 )
            ai, bi = coeffsLine
            yn = y/bi
            an = ai/bi
            bn = 1.0
            pi = np.poly1d( [an, bn ] )
            yi = pi( -x[:offset] )
            yi = yi[::-1]
            yn2 = np.zeros( len(y) + offset )
            yn2[:offset] = yi
            yn2[offset:] = yn
            xn2 = np.arange( len(yn2) ) /fph - offset/fph
            #xn2a, yn2a = getSmoothData2( xn2, yn2, smoothNumA, outlier = True )
            xn2a, yn2a = getSmoothData2( x, y, smoothNumA, outlier = True )
            x2, y2, x3, y3, x4, y4 = myGradient(xn2a, yn2a, lenAG1, lenAG2, lenAG3 )
            plt.figure()
            plt.plot(x2, y2, "ro-" )
            plt.plot(x3, y3, "b-" )
            plt.plot( [0,0], [0, max(y2)], "g-" )
            plt.plot([0, x[-1] ], [0, 0], "g-")
            plt.savefig("logs2/area/area_Plot_%003d_grad" %ii, dpi=100)
            plt.close()
            plt.figure()
            plt.plot(x3, y3, "bo-" )
            plt.plot(x4, y4, "k-" )
            plt.plot([0, x[-1] ], [0, 0], "g-")
            plt.savefig("logs2/area/area_Plot_%003d_grad2" %ii, dpi=100)
            plt.close()
            
            log.write(str(ii)+"\r\n")
            extremes0 = getExtremes( xn2, yn2, x2, y2 )
            log.write( "orig. data: "+str(extremes0)+"\r\n" )
            extremes1 = getExtremes( x2, y2, x3, y3 )
            log.write( "first der.: "+str(extremes1)+"\r\n" )
            extremes2 = getExtremes( x3, y3, x4, y4 )
            log.write( "second der.: "+str(extremes2)+"\r\n" )
            log.write("\r\n")
            
        log.close()
    
    if createAreaPlots == True:
        print areaAr.shape
        for ii in xrange(d2):
            y = areaAr[:, ii]
            x1, y1 = getSmoothData2(x, y, smoothNumA, outlier = True )
            plt.figure()
            plt.plot(x, y, "ko-" )
            plt.plot(x1, y1, "b-" )
            plt.savefig("logs2/area/area_Plot_%003d" %ii, dpi=100)
            plt.clf
            plt.close()
            
    if createAreaDetail == True:
        logD = open("logs2/area/extremes_areaD.txt", "w")
        for ii in xrange(d2):
            detailId = detailA * fph
            y = areaAr[0:detailId, ii]
            xad = x[0:detailId]
            x1, y1 = getSmoothData2( xad, y, smoothNumAD, outlier = False )
            yp, rSquared, coeffs = dataPolyFit( x1, y1, 3, returnCoeffs = True )
            ex1, ex2, xInf = extremsFromPoly3( coeffs )
            exLog = []
            if ex1:
                #if (ex1[1] > 0.5) and (ex1[1] < 2.0):
                if (ex1[1] > 0.5) and (ex1[1] < 5.0):
                    exLog.append(ex1)
            if ex2:
                #if (ex2[1] > 0.5) and (ex2[1] < 2.0):
                if (ex2[1] > 0.5) and (ex2[1] < 5.0):
                    exLog.append(ex2)
            if xInf:
                #if xInf > 0.5 and xInf < 2.0:
                if xInf > 0.5 and xInf < 5.0:
                    exLog.append(xInf)
                
            logD.write(str(ii)+"\r\n")
            logD.write(str(rSquared)+"\r\n")
            logD.write(str(exLog)+"\r\n")
            plt.figure()
            plt.plot(xad, y, "ko-" )
            plt.plot(x1, y1, "b-" )
            plt.plot(x1, yp, "r-" )
            plt.savefig("logs2/area/area_Plot_%003d_det" %ii, dpi=100)
            plt.clf
            plt.close()
            
        logD.close()
        
    if areaAnalyse2 == True:
        gridK1 = GRID1
        gridK2 = GRID2
        gridF = GRIDF
        logFit = open("logs2/area/log_fit.txt", "w")
        logFit.write("number seed, value, S0, a, k1, k2, d, a0\r\n")
        logFitF = open("logs2/area/log_fitF.txt", "w")
        logFitF.write("number seed, value, S0, a, k1, k2, d, a0\r\n")
        print areaAr.shape
            
        for ii in xrange(d2):
            if ii in [ 10, 12, 16 ]:
                ##gridK1 = ( 0, 10.0, 10001 )
                ##gridK2 = ( 0, 10.0, 10001 )
                continue
            y = areaAr[:, ii]
            #sigma = np.ones( y.shape )[300:] = 2.0
            y = y[0:300]
            x = x[0:300]
            diffYYpNormAr, coeffAr = fitModel5( x, y, ii, gridK1, gridK2, sigma = None, oneSide = True )
            #np.savetxt("diffYYpNormAr", diffYYpNormAr)
            #np.savetxt("coeffAr", coeffAr)
            
            #diffYYpNormArA, diffYYpNormArB = divideMatrix( diffYYpNormAr )
            #idCoeffA = np.nanargmin(diffYYpNormArA)
            #idCoeffB = np.nanargmin(diffYYpNormArB)
            #idCoeffA = np.unravel_index( idCoeffA, diffYYpNormAr.shape )
            #idCoeffB = np.unravel_index( idCoeffB, diffYYpNormAr.shape )
            #S0A, aA, k1A, k2A, dA, a0A = coeffAr[idCoeffA]
            #S0B, aB, k1B, k2B, dB, a0B = coeffAr[idCoeffB]
            #ypA = model5( x, S0A, aA, k1A, k2A, dA, a0A )
            #ypB = model5( x, S0B, aB, k1B, k2B, dB, a0B )
            
            idCoeff = np.nanargmin( diffYYpNormAr )
            idCoeff = np.unravel_index( idCoeff, diffYYpNormAr.shape )
            print "abs min: ", idCoeff, diffYYpNormAr[idCoeff]
            localMinList, absMin, minValueList = getLocalMin( diffYYpNormAr )
            if absMin is None:
                print "The local minimum was not found!"
                logFit.write("%d, No local minimum! The minimum value: %f in (%d, %d)\r\n" %( ii, diffYYpNormAr[idCoeff], idCoeff[0], idCoeff[1] ) )
                logFit.flush()
                continue
            
            idCoeff = absMin
            print "localMinList, absMin, minValueList", localMinList, absMin, minValueList
            S0, a, k1, k2, d, a0 = coeffAr[idCoeff]
            
            #logFit.write("%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeffA], S0A, aA, k1A, k2A, dA, a0A ) )
            #logFit.write("%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeffB], S0B, aB, k1B, k2B, dB, a0B ) )
            logFit.write("%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeff], S0, a, k1, k2, d, a0 ) )
            logFit.flush()
            for item in localMinList:
                item = tuple(item)
                print "Coeficints: ", item, diffYYpNormAr[item], coeffAr[item]
                logFit.write("Coeficints: "+str( item )+", "+str(diffYYpNormAr[item])+", "+str(coeffAr[item])+"\r\n" )
            logFit.flush()
            
            print "Seed number, diffYYpNorm", ii, diffYYpNormAr[idCoeff]
            
            plt.figure()
            xAr = np.linspace( gridK2[0], gridK2[1], gridK2[2] )
            yAr = np.linspace( gridK1[0], gridK1[1], gridK1[2] )
            plt.contourf( xAr, yAr, diffYYpNormAr, 200 )
            plt.savefig("logs2/area/tmp/diffYYp_%003d" %ii, dpi=500)
            plt.clf
            plt.close()
            
            maxValueIm = diffYYpNormAr[idCoeff] + 100.0
            dataForDetail = diffYYpNormAr
            dataForDetail[ dataForDetail > maxValueIm ] = np.nan
            plt.contourf( xAr, yAr, dataForDetail, 100 )
            plt.savefig("logs2/area/tmp/diffYYp_%003d_D" %ii, dpi=500)
            plt.clf
            plt.close()
            
            #if k1A != k2B:
                #print "!!!k1A != k2B!!!"
                #print "%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeffA], S0A, aA, k1A, k2A, dA, a0A )
                #print "%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeffB], S0B, aB, k1B, k2B, dB, a0B )
                #continue
                
            #if k1A == gridK1[1]:
                #gridK12 = gridK1
                #k1A2 = k1A
                #while k1A2 == gridK12[1]:
                    #gridK12 = ( k1A, k1A2 + 1.0, 1001 )
                    #diffYYpNormAr2, coeffAr2 = fitModel5( x, y, ii, gridK12, gridK2 )
                    #idCoeffA2 = np.nanargmin(diffYYpNormAr2)
                    #idCoeffA2 = np.unravel_index( idCoeffA2, diffYYpNormAr2.shape )
                    #S0A2, aA2, k1A2, k2A2, dA2, a0A2 = coeffAr2[idCoeffA2]
                    #logFit.write("%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr2[idCoeffA2], S0A2, aA2, k1A2, k2A2, dA2, a0A2 ) )
                    #print "+",ii, diffYYpNormAr2[idCoeffA2]
                    #logFit.flush()
                #k1 = k1A2
                #k2 = k2A2
            #else:
                #k1 = k1A
                #k2 = k2A
            
            gridD1F = 2.0*( ( gridK1[1] + gridK1[0] ) / ( gridK1[2] - 1 ) )
            gridD2F = 2.0*( ( gridK2[1] + gridK2[0] ) / ( gridK2[2] - 1 ) )
            gridK1F = ( k1 - gridD1F, k1 + gridD1F, gridF )
            gridK2F = ( k2 - gridD2F, k2 + gridD2F, gridF )
            print "Dedail"
            diffYYpNormArF, coeffArF = fitModel5( x, y, ii, gridK1F, gridK2F, sigma = None, oneSide = True )
            idCoeffF = np.nanargmin(diffYYpNormArF)
            idCoeffF = np.unravel_index( idCoeffF, diffYYpNormArF.shape )
            S0, a, k1, k2, d, a0 = coeffArF[idCoeffF]
            logFitF.write("%d, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormArF[idCoeffF], S0, a, k1, k2, d, a0 ) )
            print "Seed number, diffYYpNormF", ii, diffYYpNormArF[idCoeffF]
            logFitF.flush()
            
            yp = model5( x, S0, a, k1, k2, d, a0 )
            plt.figure()
            plt.plot(x,y,"k")
            plt.plot(x,yp,"r")
            plt.savefig( "logs2/area/area_Plot_fit_%003d-F" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            ypDI = model5DI( x, a, k1, k2, d, a0 )
            ypDII = model5DII( x, a, k1, k2, d )
            plt.figure()
            plt.plot(x,ypDI,"r")
            plt.plot(x,ypDII,"b")
            plt.savefig( "logs2/area/area_Plot_fit_%003d-FD" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            #initEs5 = np.array([ y[0], 2000.0, 0.2, 0.1, 0.0, 1000.0 ])
            ##sigma = None
            #sigma = np.linspace(1.0, 2.0, len(y) )
            #try:
                #popt5, pcov5 = optimize.curve_fit( model5, x, y, initEs5 )
            #except:
                #print ii
                #print "problem... TODO"
                #continue
                
            #print ii
            
            ##model5
            #S0, a, k1, k2, d, a0 = popt5
            #print "S0, a, k1, k2, d, a0, k0", S0, a, k1, k2, d, a0
            #yp = model5( x, S0, a, k1, k2, d, a0 )
            ##ys1 = model5S1( x, S0, a, k1, k2, d )
            ##ys2 = model5S2( x, S0, a, k1, k2, d)
            ##ypI = model5DI( x, a, k1, k2, d )
            ##ypII = model4DII( x, a, k1, c, k2 )
            
            #plt.figure()
            #plt.plot(x[:], y[:], "k-" )
            #plt.plot( x, yp, "r-" )
            #plt.savefig("logs2/area/area_Plot_%003d_exp1" %ii, dpi=100)
            #plt.clf
            #plt.close()
            
            
        sys.exit()
    
    kk = 0
    for item in efdsCoeff:
        plotLimits = [ np.min( efds[:, :, item] ), np.max( efds[:, :, item] ) ]
        dataOutP = dataOutPut[kk]
        if createEfdsPlotsOrig == True:
            for ii in xrange(d2):
                y = efds[:, ii, item]
                plt.figure()
                plt.plot( x, y, "ko-")
                plt.ylim( plotLimits[0], plotLimits[1] )
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d" %ii, dpi=100)
                plt.clf
                plt.close()
                
        if createEfdsPlotsGrad1 == True:
            #length1 = 12
            #length2 = 12
            for ii in xrange(d2):
                y = efds[:, ii, item ]
                
                x1, y1 = getSmoothData2( x, y, smoothNumA, outlier = True )
                x2, y2, x3, y3, x4, y4 = myGradient(x1, y1, lenG1, lenG2, lenG3 )
                plt.figure()
                plt.plot(x2, y2, "ro-" )
                plt.plot(x3, y3, "b-" )
                plt.plot([0, n/fph], [0, 0], "g-")
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_grad" %ii, dpi=100)
                plt.close()
                plt.figure()
                plt.plot(x3, y3, "bo-" )
                plt.plot(x4, y4, "k-" )
                plt.plot([0, n/fph], [0, 0], "g-")
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_grad2" %ii, dpi=100)
                plt.close()
                
            
        if createEfdsPlotsGrad2 == True:
            length1 = 24
            length2 = 12
            for ii in xrange(d2):
                y = efds[:, ii, item ]
                plt.figure()
                x1, y1, x2, y2, x3, y3 = myGradient( x, y, length1, length2 )
                x1 = x1 * 1/fph
                x2 = x2 * 1/fph
                #print x1
                plt.plot( x1, y1, "ro-")
                plt.plot( x2, y2, "b-")
                plt.plot([0, n/fph], [0, 0], "g-")
                
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_grad24-12" %ii, dpi=100)
                plt.clf
                plt.close()
                
        if createEfdsPlotsPoly == True:
            degree = DEGREE
            for ii in xrange(d2):
                y = efds[:, ii, item]
                x1, y1 = getSmoothData2( x, y, smoothNumA, outlier = True )
                plt.figure()
                plt.plot( x, y, "ko-")
                plt.plot( x1, y1, "b-")
                yp, coefficients = dataPolyFit( x1, y1, degree )
                plt.plot( x1, yp, "r-")
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_poly" %ii, dpi=100)
                plt.clf
                plt.close()
        
        if createEfdsPlotsPoly2 == True:
            degree = 4
            for ii in xrange(d2):
                y = efds[:, ii, item]
                plt.figure()
                plt.plot( x, y, "ko-")
                yp, coefficients = dataPolyFit( x, y, degree )
                plt.plot( x, yp, "r-")
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_poly2" %ii, dpi=100)
                plt.clf
                plt.close()
        
        if createEfdsPlotsPoly3 == True:
            degree = 6
            for ii in xrange(d2):
                y = efds[:, ii, item]
                plt.figure()
                plt.plot( x, y, "ko-")
                yp, coefficients = dataPolyFit( x, y, degree )
                plt.plot( x, yp, "r-")
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_Plot_%003d_poly6" %ii, dpi=100)
                plt.clf
                plt.close()
            
        kk += 1
            

if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
    
    print "ignore list", IGNORE_LIST
    directory = sys.argv[2]
    switch = sys.argv[1]
    if switch == "efd":
        efdAnalyse1( directory )
        
    print "END"
