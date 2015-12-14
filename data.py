"""
  Tools for analyse of logsfiles
    usage:
         python data.py <switch> <directory>
            switch: a - area
            switch: efd
"""

import sys
import os
import numpy as np
from matplotlib import pyplot as plt

IGNORE_LIST = [ 96, 116 ]
FRAMES_PER_H = 30.0


def getEfdCoefficient(num):
    n = num / 4
    l = num % 4
    if l == 0:
        letter = "a"
    elif l == 1:
        letter = "b"
    elif l == 2:
        letter = "c"
    elif l == 3:
        letter = "d"
        
    return str(n) + letter
    

def readData( fileList, directory, subject, ignorList ):
    if subject == "area":
        fileIdList = []
        areaList = []
        for fileN in fileList:
            #print fileN
            fId = fileN.split("_")[1]
            fId = int(fId) - 1
            fileIdList.append(fId)
            
            areaL = []
            f = open( directory+fileN, "r" )
            label = None
            for line in f:
                line = line.split(":")
                if line[0] == label:
                    label = int(line[1])
                    continue
                
                if line[0] == "area (mm2)" and label not in ignorList:
                    area = float(line[1])
                    areaL.append(area)
            
            f.close()
            areaList.append(areaL)
        
        #print fileIdList
        areaList = np.array(areaList)
        areaAr = np.zeros( (areaList.shape) )
        
        ii = 0
        for fId in fileIdList:
            areaAr[fId] = areaList[ii]
            ii += 1
        
        return areaAr
        
    elif subject == "efd":
        fileIdList = []
        totalEfdList = []
        for fileN in fileList:
            #print fileN
            fId = fileN.split("_")[1]
            fId = int(fId) - 1
            fileIdList.append(fId)
            
            efdList = []
            efdL = []
            f = open( directory+fileN, "r" )
            read = False
            label = None
            for line in f:
                lineSplit = line.split(":")
                if line[0] == label:
                    label = int(line[1])
                    continue
                
                if lineSplit[0] == "efd" and label not in ignorList:
                    read = True
                    continue
                    
                if len(line) < 4:
                    read = False
                    efdList.append(efdL)
                    efdL = []
                    continue
                
                if read == True:
                    line = line[1:-3]
                    #print line
                    numFromLine = map(float, line.split() )
                    for item in numFromLine:
                        efdL.append(item)
            
            f.close()
            
            totalEfdList.append(efdList)
            
        totalEfdList = np.array(totalEfdList)
        totalEfdArS = np.zeros( totalEfdList.shape )
        ii = 0
        for fId in fileIdList:
            totalEfdArS[fId] = totalEfdList[ii]
            ii += 1
        
        #print totalEfdList.shape
        #print totalEfdArS[144, 80, :]
        
        return totalEfdArS


def getLogs( directory ):
    fileList = os.listdir( directory )
    #print fileList
    
    cleanFileL = []
    for fileN in fileList:
        if fileN.split(".")[-1] == "txt":
            cleanFileL.append( fileN )
    
    print len(cleanFileL)
    return cleanFileL


def areaAnalyse(directory):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H 
    subject = "area"
    
    fileList = getLogs( directory )
    areaAr = readData( fileList, directory, subject, ignorList )
    #print "{0}".format(areaAr[0])
    print areaAr.shape
    
    normAreaAr = areaAr.copy()
    initArea = areaAr[0]
    #print initArea
    for ii in xrange( len(areaAr) ):
        normAreaAr[ii] = areaAr[ii] / initArea
        
    print "{0}".format(normAreaAr)
    
    n = len(normAreaAr)
    meanNormArea = np.zeros( n )
    sdNormArea = np.zeros( n )
    for ii in xrange( n ):
        meanNormArea[ii] = np.mean( normAreaAr[ii] )
        sdNormArea[ii] = np.std( normAreaAr[ii] )
    
    print meanNormArea, sdNormArea
    x = ( np.array(range( n )) + 1 ) * 1/fph
    plt.errorbar( x, meanNormArea, yerr=sdNormArea )
    plt.show()
    
    results = open("logs/result_area.txt", "w")
    results.write("note:\r\n")
    results.write("Ignore list: "+str(ignorList)+"\r\n")
    results.write("meanNormArea\r\n")
    results.write(str( list( meanNormArea ) )+"\r\n")
    results.write("sdNormArea\r\n")
    results.write(str( list( sdNormArea ) )+"\r\n")
    results.write("normAreaAr\r\n")
    for item in normAreaAr:
        results.write( str( list(item) ) )
        results.write("\r\n")
        
    results.write("areaAr\r\n")
    for item in areaAr:
        results.write( str(list(item)) )
        results.write("\r\n")

    results.close()


def efdAnalyse( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    meanEfds = np.zeros( ( efds.shape[0], efds.shape[2] ) )
    sdEfds = np.zeros( meanEfds.shape )
    varCoEfds = np.zeros( meanEfds.shape )
    ii = 0
    for efd in efds:
        meanEfds[ii] = np.mean( efd, axis = 0 )
        sdEfds[ii] = np.std( efd, axis = 0 )
        varCoEfds[ii] = sdEfds[ii] / abs(meanEfds[ii])
        ii += 1
    
    print meanEfds[0]
    print sdEfds[0]
    print varCoEfds[0]
    print meanEfds.shape
    
    #plt.hist(efds[0, :, 8])
    #plt.title("koeficient: "+str(ii))
    #plt.savefig('hists_test/Hist'+str(ii), dpi=50)
    
    for jj in range(80):
        plt.figure()
        plt.hist(efds[0, :, jj])
        print jj
        plt.title("koeficient: "+str(jj))
        plt.savefig('hists_test/Hist'+str(jj), dpi=50)
        plt.close()
    
    n = meanEfds.shape[0]
    for ii in xrange( meanEfds.shape[1] ):
        coefficient = getEfdCoefficient(ii)
        plt.figure()
        x = ( np.array(range( n )) + 1 ) * 1/fph
        plt.errorbar( x, meanEfds[:, ii], yerr=sdEfds[:, ii] )
        plt.title("coefficient: "+coefficient)
        plt.savefig("efdsPlots/efds_Plot_"+coefficient, dpi=100)
        plt.clf
        #plt.show()
        plt.close()
        
    x = range(44)
    for jj in xrange( meanEfds.shape[0] ):
        plt.figure()
        plt.plot( x, abs(meanEfds[jj, 0:44]), "ro-" )
        plt.grid( True )
        plt.axis([0, 44, 10e-6, 1.1])
        plt.yscale('log')
        time = (jj + 1) *1/fph
        plt.title("time: "+str(time) )
        plt.savefig("efdsPlots2/efds_Plot2_%003d" %jj, dpi=100)
        plt.close()
        


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        print "ignore list", IGNORE_LIST
        sys.exit(1)
    
    directory = sys.argv[2]
    switch = sys.argv[1]
    if switch == "a":
        areaAnalyse( directory )
        
    elif switch == "efd":
        efdAnalyse( directory )
        
    print "END"
