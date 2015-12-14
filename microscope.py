#!/usr/bin/python
"""
  View part of images
  usage:
       ./microscope.py <directory> [<x,y,size>]
"""
import sys
import os
import cv2

resize = 4

Xi = 1920
Yi = 2120
Xe = 2108
Ye = 2248

Xi = 2037
Yi = 2256
Xe = 2193
Ye = 2388


def microVideo( directory, xi, yi, xe, ye ):
    for filename in sorted(os.listdir( directory )):
        if filename.endswith(".JPG"):
            img = cv2.imread( os.path.join(directory, filename) )
            img2 = img[yi:ye, xi:xe]
            img3 = cv2.resize(img2, None, fx = resize, fy = resize, interpolation = cv2.INTER_CUBIC)
            cv2.imshow( "image", img3 )
            if cv2.waitKey(1000) > 0:
                
                break
            print filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(2)

    microVideo( sys.argv[1], Xi, Yi, Xe, Ye )

# vim: expandtab sw=4 ts=4 

