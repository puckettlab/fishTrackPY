#-------------------------------------------#
# main python code
# run as:
#   python3 runTracking.py data/sampleVideo.mp4 data/cameraCalib.npz
#-------------------------------------------#
import getopt, sys
import numpy as np
from videoDetect        import *            #detection functions
from videoTrack         import *            #tracking functions
from videoTrack2world   import *            #tracking functions
from videoTrackSplice   import *            #splice trajectories
from makeTrackVideo     import *            #make video
#-------------------------------------------#
# parameter to set, hard-code, can also use the cli to set these
img_threshold           = 30                # image threshold to detect
imgMinLineLength        = 3                 #
noisy                   = False             # plot/show while tracking, to help tune parameters
#-------------------------------------------#
# begin main code
#-------------------------------------------#
if __name__ == "__main__":
    args, fname         = getopt.getopt(sys.argv[1:], '', ['th=', 'noisy='])
    if len(fname)<1:
        print("not enough input arguments")
        print("Usage:  python runTracking.py data/sampleVideo.mp4")
        print("  or if a camera calibartion file is available, then")
        print("Usage:  python runTracking.py data/sampleVideo.mp4 data/cameraCalib.npz\n")
        sys.exit()
    #-------------
    vname               = fname[0]
    if len(fname)<2:                            #if no cameraCalib given, don't use, everything in pixels
        camData         = None                  #don't use a camera calibration
    else:
        cameraFname     = fname[1]              #cameraCalib filename
        camData         = np.load(cameraFname)  #load
    #-------------
    print(camData['K'])
    print("Detecting particles")
    videoDetect(vname,img_threshold,imgMinLineLength,noisy,camData)
    #-------------
    print(vname)
    print("Finished detecting...\nTracking particles")
    videoTrack(vname,noisy,camData)
    ##-------------
    print(vname)
    print("Finished tracking...\nConverting tracks to world units")
    videoTrack2world(vname,noisy,camData)       #img 2 world coordinates for tracks
    ##--------------
    print("make track video\n\n")
    makeTrackVideo(vname,noisy,camData)       #img 2 world coordinates for tracks
