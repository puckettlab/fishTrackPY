# Python 2/3 compatibility
from __future__ import print_function
#import argparse
import sys
import os.path
import getopt
from glob import glob
import yaml
from videoDetect    import * 		#detection functions
from videoTrack     import * 		#tracking functions
from videoTrack2world import * 		#tracking functions
#import cv2,numpy 					#required
#-------------------------------------------#

#-------------------------------------------#
threshold_Default 	= 30
noisy_Default 		= False
qReRun 			= True
#cname 				= "14135929" 			#camera name

	
	
if __name__ == "__main__":
    args, fname0 = getopt.getopt(sys.argv[1:], '', ['th=', 'noisy='])
    vname = fname0[0]
    args = dict(args)
    args.setdefault('--th', threshold_Default)
    args.setdefault('--noisy', noisy_Default)
    noisy 		= int(args.get('--noisy'))
    img_threshold	= int(args.get('--th'))
    filetype 	= 'mp4'
    #			#filetype
    #vname  = baseDir/runname/runname.mp4 
    fDir, fName 	= os.path.split(vname) 	 			#use video path to find runname, base directory
    baseDir, runname= os.path.split(fDir) 	 			#base directory, runname
    #=====================
    #get camera calibration
    #
    #-------
    #old version to get camera
    ###cameraFname 	= baseDir+'/calibration/'+cname+'.npz'
    #-------
    cFname 		= glob(baseDir+'/201*calib.npz')
    cameraFname 	= cFname[0]
    camData 		= np.load(cameraFname)
    print("Getting background")
    getBG(vname,noisy)
    #-------------
    imgMinLineLength = 2		
    imgLineMultiply  = 1
    print("Detecting particles")
    print(vname)
    dname = vname[:-4]+'-detect.h5'
    if ((not os.path.exists(dname)) or (qReRun)):
	    videoDetect(vname,camData,img_threshold,imgMinLineLength,imgLineMultiply,noisy)
    #-------------
    print("Finished detecting...\nTracking particles")
    print(vname)
    videoTrack(vname,camData,noisy)
    print(vname)
    print("Finished tracking...\nConverting tracks to world units")
    videoTrack2world(vname,camData,noisy) 	#img 2 world coordinates for tracks
    #-------------
