# Python 2/3 compatibility
import cv2
import sys
import os.path
from cameraFunctions import *
from scipy.signal import savgol_filter
#-------------------------------------------#
#-------------------------------------------#
def videoTrack2world(vname,noisy=0,camData=None):
    porder      = 3         ##order of scipy.signal.savgol_filter
    winLength   = 11
    #
    loadName    = vname[:-4]+"-track.npz"
    saveName    = vname[:-4]+"-vtracks.npz"             #tracks in World coordinates
    if ( not os.path.isfile(loadName) ):
        print ("Looking for file: %s"%loadName)
        print ("No detect file found")
        sys.exit()
    ##
    data        = np.load(loadName);
    fps         = data['fps']
    uv          = data['x'] ##xy in pixel coords
    t           = data['t']
    ind         = data['ind']
    #data has shape [t, ind, x, y]
    xyzw        = np.zeros((t.shape[0],3))      #xyz
    ut          = np.unique(t)
    uind        = np.unique(ind)
    #----
    dataX       = np.zeros((0,3))       #position
    dataV       = np.zeros((0,3))       #velocity
    dataA       = np.zeros((0,3))       #acceleration
    dataT       = np.zeros((0,))        #time
    dataI       = np.zeros((0,))        #index
    #----
    if camData is not None:
        cam_K       = camData['K']
        cam_dist    = camData['dist']
        cam_R       = camData['R']
        cam_tvec    = camData['tvec']
    #----
    # loop over trajectories
    # calculate velocity/acceleration. convert to world units if camData is included
    for indi in uind:
        inn     = ind==indi             #
        if (np.sum(inn)> (winLength) ):
            uvi         = uv[inn,:]
            ti          = t[inn]
            if camData is not None:
                xyzi    = image2world(uvi,cam_K,cam_R,cam_tvec,cam_dist)
            else:
                xyzi    = np.hstack( (uvi,np.ones((uvi.shape[0],1))) )
            #---
            vxi         = savgol_filter(xyzi[:,0], window_length=winLength, polyorder=porder, deriv=1)*fps
            vyi         = savgol_filter(xyzi[:,1], window_length=winLength, polyorder=porder, deriv=1)*fps
            vzi         = savgol_filter(xyzi[:,2], window_length=winLength, polyorder=porder, deriv=1)*fps
            #---
            axi         = savgol_filter(xyzi[:,0], window_length=winLength, polyorder=porder, deriv=2)*fps*fps
            ayi         = savgol_filter(xyzi[:,1], window_length=winLength, polyorder=porder, deriv=2)*fps*fps
            azi         = savgol_filter(xyzi[:,2], window_length=winLength, polyorder=porder, deriv=2)*fps*fps
            #---
            vvi         = np.vstack((vxi,vyi,vzi)).T
            aai         = np.vstack((axi,ayi,azi)).T
            #---
            dataX       = np.vstack((dataX,xyzi))
            dataV       = np.vstack((dataV,vvi))
            dataA       = np.vstack((dataA,aai))
            dataT       = np.hstack((dataT,ti.astype('float')/fps))                            #time in frames to time in seconds
            dataI       = np.hstack((dataI,indi*np.ones((ti.shape[0],))))
            #---
    #------------------------------
    print(dataX.shape)
    np.savez_compressed(saveName,x=dataX,v=dataV,a=dataA,t=dataT,ind=dataI)
    print("finished converting to world coordinates")
