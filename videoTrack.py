# Python 2/3 compatibility

import cv2
import sys
from scipy.optimize import linear_sum_assignment #hungarian algorithm
from trackClass import Track
####import h5py
import time
from videoFunctions import *
from cameraFunctions import *
#-------------------------------------------#
#matching parameters
maxDistanceKF   = 20                        # if distance to new center < this, match! this is distance squared
maxNotFoundKF   = 5                         # kalman patch track; if found in 5 frames add new track
thErrorMax      = np.pi/8.;                 # radians
#-------------------------------------------#
imod            = 20;                       # for printing time
noisy_default   = 0                         #no plotting/visual
#-------------------------------------------#
#-------------------------------------------#
def getErrorDistance(x1,x2):
    D = np.zeros( ( x1.shape[0] , x2.shape[0]) )    #current,old
    for i in range(x1.shape[0]):
        for j in range( x2.shape[0] ):
            dx = (x1[i,0] - x2[j,0])
            dy = (x1[i,1] - x2[j,1])
            D[i,j] = (dx*dx + dy*dy)**0.5
    return D
#-------------------------------------------#
def getErrorAngle(x1,x2):
    D = np.zeros( ( x1.shape[0] , x2.shape[0]) )    #current,old
    for i in range(x1.shape[0]):
        for j in range( x2.shape[0] ):
            dth = (x1[i] - x2[j])                                   #unwrapped!!!!!
            dth = (dth + np.pi/2.) % np.pi - np.pi/2.               ###dth = (dth + np.pi) % 2*np.pi - np.pi        #in radians,
            D[i,j] = np.abs(dth)
    return D
#-------------------------------------------#
def getDataTrack(tracks,t):
    data = np.empty((tracks.shape[0],4))    #t,ind,x,y
    keep = np.ones((tracks.shape[0]))>0     #t,ind,x,y
    for (i,f) in enumerate(tracks):
        if f.qVisible:                                          #save the visible too???
            data[i,0] = t
            data[i,1] = f.ind
            data[i,2] = f.x[0,0]
            data[i,3] = f.x[0,1]                    #more??? or prediction???
        else:
            keep[i]         = False
    data = data[keep,:]
    return data
#-------------------------------------------#
def plotTrack(img,tracks):
    out                     = img.copy()
    qPlotHist               = True
    nPlotHistLength         = 30
    qPlotNumber             = False
    for i in range(tracks.shape[0]):
        f = tracks[i]
        if f.qVisible:
            cv2.circle(out,(int(f.x[0,0]),int(f.x[0,1])),5,f.color,-20)
            ind_str = (str)(f.ind)
            if (qPlotNumber):
                cv2.putText(out,ind_str, (int(f.x[0,0]),int(f.x[0,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            if ( (f.xhist.shape[0]>2) and qPlotHist ):
                rev_hist = f.xhist[::-1]
                for j in range(f.xhist.shape[0]-1):
                    xp1 = ( (int) (rev_hist[j,0]),          (int) (rev_hist[j,1]) )
                    xp2 = ( (int) (rev_hist[(j+1),0]), (int) (rev_hist[(j+1),1]) )
                    cv2.line(out, xp1,xp2, f.color);
                    if (abs(j)>nPlotHistLength):    #only last 5 frames
                        break
    return out
#-------------------------------------------#
def initializeTrack(data):
    N0 = data.shape[0]
    tracks = np.ndarray(( N0,),dtype=np.object)                                     #list of matched track/contours
    for i in range( N0 ):   #each contour
        x               = Track()
        x.setPosition(data[i][1],data[i][2],data[i][3])                         #[t, cX, cY, angle, area, MA, ma]
        tracks[i] = x
    return tracks

#-------------------------------------------#
def matchTrack(tracks, data):
    xpred   = [f.pred for f in tracks]                                              #use distance of KF cost matrix for matching
    xpred   = np.reshape(xpred,(tracks.shape[0],2))
    Dx      = getErrorDistance(xpred,data[:,[1,2]] )                #Distance from KF prep to contours
    row_dist, col_dist = linear_sum_assignment(Dx)                  #pos, column is the matched id; fit with hungarian algorithm
    #-----
    thpred  = np.array( [f.pred_th for f in tracks] )               #np.array( [f.th for f in tracks] )
    Dt      = getErrorAngle(thpred, data[:,3] )
    #-----
    wD      = (0.5, 0.1 )
    Dfull   = wD[0]*Dx/maxDistanceKF + wD[1]*Dt/thErrorMax
    row_dist, col_dist = linear_sum_assignment(Dfull)
    #----
    idNew   = np.arange( data.shape[0] )                                    #helper list of un-matched contours, current frame
    idOut   = []
    #----
    for i in range( tracks.shape[0] ):
        qGood       = False
        innd        = i == row_dist
        if (np.sum(innd)>0 ):                                                           #this row was used
            id      = np.where(innd)
            idi     = col_dist[id]
            idi     = idi[0]
            if (Dx[i,idi] < maxDistanceKF):
                qGood = True
        #----
        if (qGood):
            tracks[i].setPosition(data[idi][1],data[idi][2],data[idi][3])
            idNew = idNew[idNew!=idi]
        else:
            tracks[i].setNotFound() #
            if (tracks[i].kf_notFound>maxNotFoundKF):
                idOut.append(i)
    #-----------------------
    for i in reversed(idOut):
        tracks          = np.delete(tracks,i)
    #new track, worry about track splice later
    for i in idNew:
        x               = Track()
        x.setPosition(data[i][1],data[i][2],data[i][3])
        tracks = np.append(tracks,x)
    return tracks


#-------------------------------------------# main method
def videoTrack(vname,noisy=0,camData=None):
    saveName                = vname[:-4]+"-track.npz"
    loadName                = vname[:-4]+"-detect.npz"
    if ( not os.path.isfile(loadName) ):
        print ("Looking for file: %s"%loadName)
        print ("No detect file found")
        sys.exit()
    #----
    data                    = np.load(loadName);
    data                    = data['sData']
    #----
    bg                      = getBG(vname)                                  # load (or create) bg image from video
    if camData is not None:
        bg                  = cv2.remap(bg,camData['mapx'],camData['mapy'],cv2.INTER_LINEAR) #undistort
    #----
    cap                     = cv2.VideoCapture(vname)               # load video
    Nframes                 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps                     = cap.get(cv2.CAP_PROP_FPS)
    #----
    if ( noisy==0 ):
        cap.release()       # then don't need video any more
    saveData                = np.empty((0,5))                               #t,ind,x,y,z
    #----
    tstart                  = time.time()
    Ni                      = 0;
    tracks                  = []
    x                       = []
    ind                     = []
    t                       = []
    #---
    for i in range(Nframes):
        ini         = data[:,0].astype(int) == (i+1)
        datai       = data[ini,:]                                   #data from this frame
        if (i==0):                                                              #initialize tracks
            tracks          = initializeTrack(datai) #
        else:
            tracks          = matchTrack(tracks,datai)
        #----
        dataNow             = getDataTrack(tracks,i)
        t.append(dataNow[:,0])
        ind.append(dataNow[:,1])
        x.append(dataNow[:,[2,3]])
        Ni = Ni+dataNow.shape[0]                #only printing to check progress
        if ( noisy>0 ):
            ret, frame      = cap.read()
            gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray            = cv2.remap(gray,camData['mapx'],camData['mapy'],cv2.INTER_LINEAR)# undistort
            frame2          = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            out             = plotTrack(frame2,tracks)
            qSaveVideo      = 1
            if (qSaveVideo):
                outname = "tmp/"+vname[:-4]+"-track+%04d.png"%i #video
                cv2.imwrite(outname,out)             #video
            out = cv2.resize(out, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            cv2.namedWindow("frame")
            cv2.imshow('frame',out)
            cv2.waitKey(10)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                sys.exit()
        #----
        if (i%imod ==0 and i>0):
            dt      = time.time() - tstart
            print ("%04.0f Nfound_ave=%3.1f  te=%3.3f tr=%3.3f"%(i,(float)(Ni)/imod, dt, dt/(i+1)*(Nframes-i)))
            Ni      = 0;
        #----
    ind         = np.concatenate(ind).ravel().astype(int)
    t           = np.concatenate(t).ravel().astype(int)
    x           = np.reshape(np.concatenate(x).ravel(), (-1,2))
    [t,ind,x]   = removeShortTracks(t,ind,x)
    [t,ind,x]   = interpolateTracks(t,ind,x)
    saveData    = np.vstack((t,ind,x.T)).T            #t, cx, cy, ang, headx, heady,headint
    #------------------------------
    # prepare/save data
    np.savez_compressed(saveName,t=t,ind=ind,x=x,fps=fps)
    print (saveData.shape)
    print("finished with tracking")

if __name__ == "__main__":
    main(sys.argv)
