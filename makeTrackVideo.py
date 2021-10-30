# Python 2/3 compatibility
import cv2
import sys
import os
import time
from matplotlib import cm
from videoFunctions import *
from cameraFunctions import *
#-------------------------------------------#
def plotTrackZ(img,ut,t,uv,ind,color):
    out             = img.copy()
    qPlotHist       = True
    dt              = 30            #history line length
    lw              = 1             #history line width
    #----
    innt            = (t>ut-dt)*(t<=ut)
    uvt             = uv[innt]
    indt            = ind[innt]
    tt              = ind[innt]
    uind            = np.unique(indt)
    #----
    for i in range(uind.shape[0]):
        inn         = indt==uind[i]
        uvti        = uvt[inn]
        uvtc        = uvti[-1]
        cv2.circle(out,(int(uvtc[0]),int(uvtc[1])),5,color[uind[i]],-20)
        if (qPlotHist and uvti.shape[0]>1):
            for j in range(uvti.shape[0]-1):
                xp1 = ( (int) (uvti[j,0]),     (int) (uvti[j,1]) )
                xp2 = ( (int) (uvti[(j+1),0]), (int) (uvti[(j+1),1]) )
                cv2.line(out, xp1,xp2, color[uind[i]],lw);
        #----
    return out
#-------------------------------------------#
def makeTrackVideo(vname,noisy=0,camData=None):
    loadName                = vname[:-4]+"-track.npz"
    if ( not os.path.isfile(loadName) ):
        print ("Looking for file: %s   \\ not found"%loadName)
        sys.exit()
    #----
    imod        = 100
    tstart      = time.time()
    #----
    data        = np.load(loadName);
    fps         = data['fps']
    uv          = data['x'] ##xy in pixel coords
    t           = data['t']
    ind         = data['ind']
    #----
    ut          = np.unique(t)
    #---- generate random colors
    uind        = np.unique(ind)
    perm        = np.arange(np.max(uind)+1)
    np.random.shuffle(perm)
    color       = cm.plasma(np.linspace(0, 1, perm.shape[0]))# define colormap
    color       = color[perm]*255
    color       = color[:,:3]
    color       = color[:,::-1]
    #----
    cap         = cv2.VideoCapture(vname)                   # load video
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # float `width`
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float `height`
    Nframes     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    #---- #create output file
    saveVideoName = loadName[:-4]+".mp4"
    fourcc      = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    outVideo    = cv2.VideoWriter(saveVideoName, fourcc, fps, (width,height)) #output video to draw tracks
    #----
    for i in range(ut.shape[0]):
        inn             = t==ut[i]
        ret, frame      = cap.read()
        if ret:
            if camData is not None:
                gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray            = cv2.remap(gray,camData['mapx'],camData['mapy'],cv2.INTER_LINEAR)# undistort
                frame           = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)     #convert to color
            frameOut        = plotTrackZ(frame,ut[i],t,uv,ind,color)
            outVideo.write(frameOut)
            if (i%imod ==0 and i>0):
                dt      = time.time() - tstart
                print("%04.0f   te=%3.3f tr=%3.3f"%(i, dt, dt/(i+1)*(Nframes-i)))
    #---- clean up
    cap.release()
    outVideo.release()
    print("finished")
    #----
#-------------------------------------------#
if __name__ == "__main__":
    main(sys.argv)
