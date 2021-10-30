import numpy as np
import cv2
import os.path
from scipy.interpolate import interp1d,splrep, splev #interpolation
minTrackLength = 5                      #need at least this many frames in a track
qReRun = 1
#-------------------------------------------#
# showImageHalfSize
#       shows image at half resolution
#-------------------------------------------#
def showImageHalfSize(img):
        img2 = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('frame',img2)
        cv2.waitKey(0)
#-------------------------------------------#
# getBG
#       load if exists otherwise call averageVideo
#-------------------------------------------#
def getBG(vname,qReRun=1,noisy=0):
    skip_frame  = 10;
    n_frame     = 40*30;  #sec * fps = 1200 frames;
    bgname              = vname[:-4]+"-bg.jpg"
    #if ((not os.path.isfile(bgname)) or (qReRun)):
    averageVideo(vname,skip_frame,n_frame,noisy)
    bg                  = cv2.imread(bgname)
    print((bg.shape))
    bg          = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)          #extract only 1 channel
    if (noisy):                                         # Display the resulting frame
            showImageHalfSize(bg)
    return bg
#-------------------------------------------#
# average Video
#       get video BG by averaging over all frames
#-------------------------------------------#
def averageVideo(vname,skip_frame=10,n_frames=120,noisy=0):
    #skip_frame = 5                                             # only compute bg every nth frame; 30fps, so twice/s
    cap         = cv2.VideoCapture(vname)               # load video
    fps         = cap.get(cv2.CAP_PROP_FPS)             #
    Nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    img         = np.zeros((int(height),int(width)),dtype=np.float)
    cnt         = 0
    Nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (Nframes>n_frames):
        Nframes = n_frames
    for i in range(Nframes):
        ret, frame = cap.read()
        if (i%skip_frame == 0):                         #only operate every nth
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img  = img+gray
            cnt  += 1
            if (noisy):                     # Display the resulting frame
                gray = cv2.resize(gray,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                cv2.imshow('frame',gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    img = img/cnt
    imgname = vname[:-4]+"-bg.jpg"
    cv2.imwrite(imgname,img)
    #clean-up; rewind
    cap.set(1,0)
    cap.release
    if (noisy):
        cv2.destroyAllWindows()


#-------------------------------------------#
def removeShortTracks(t,ind,x,minLength=minTrackLength):
        uind    = np.unique(ind)
        trLen   = np.array([np.sum(i==ind) for i in uind])
        out     = trLen < minLength
        uindOut = uind[out]
        keep    = np.ones(t.shape)>0
        for i in uindOut:
                keep[ind==i] = False
        t               = t[keep]
        ind     = ind[keep]
        x               = x[keep,:]
        return t,ind,x
#-------------------------------------------#
def interpolateTracks(t,ind,x):
        uind    = np.unique(ind)
        x1              = []
        ind1    = []
        t1              = []
        for ui in uind:
                inn     = ind==ui                                       #
                ti              = t[inn]                                        #
                if (np.sum(inn) < ti[-1]-ti[0]):        #need to interpolate
                        fx              = interp1d(ti,x[inn,0])
                        fy              = interp1d(ti,x[inn,1])
                        tinterp = np.arange(ti[0],ti[-1]+1)
                        xnew    = fx(tinterp)
                        ynew    = fy(tinterp)
                        indinterp = ui*np.ones(xnew.shape[0])
                        #store
                        ind1.append(indinterp)
                        t1.append(tinterp)
                        x1.append(np.vstack((xnew,ynew)).T)
                else:
                        ind1.append(ind[inn])
                        t1.append(t[inn])
                        x1.append(x[inn,:])
        ind1    = np.concatenate(ind1).ravel().astype(int)
        t1      = np.concatenate(t1).ravel().astype(int)
        x1              = np.reshape(np.concatenate(x1).ravel(), (-1,2))
        return t1,ind1,x1




#-------------------------------------------#
#-------------------------------------------#
#-------------------------------------------#
#misc functions for plotting with opencv
#-----
def plotContours(img,contours):
        out = img.copy()
        for c in contours:
                color  = (20+235*np.random.rand(), 20+235*np.random.rand(), 20+235*np.random.rand())
                cv2.drawContours(out,[c],-1,color,2)
        return out
