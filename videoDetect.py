# Python 2/3 compatibility
import cv2
from videoFunctions import *
from cameraFunctions import *
import sys
import time
import matplotlib.pyplot as plt
import multiprocessing                                          # parallel
#-------------------------------------------#
noisyTh                 = False                                 # show center and lines, useful for setting thresholds for imgMinLineLength
#-------------------------------------------#
th_distMid              = 20.                                   # group line segments, distant between midpoints
th_angle                = 0.80                                  # cos(theta) = 0.9, angle between line segments
th_maxZeroGroupLineImg  = 5                                     # 4-5; too low will exclude some good matches, too high will allow bad grouped lines
#minLineLength  = 22                                            # group line must be bigger than this
kernel                  = np.ones((3, 3), np.uint8)             # for closing bw
#===========================================#
imod                    = 200;                                  # for printing time
thMergeDistance         = 10;                                   # if using imgLineMultiply, if no distance is less than this, add the contour pt
#-------------------------------------------#
def getMidpoints(lines):
    lineMidPts = np.array( [ [(a[0,0]+a[0,2])/2,(a[0,1]+a[0,3])/2] for a in lines] )
    return lineMidPts
#-------------------------------------------#
def preProcessImage(gray,bg,imgThresh=10,noisy=0):
    gaussfilterSize = 5
    gray2           = cv2.subtract(bg,gray)
    gray3           = cv2.GaussianBlur(gray2,(gaussfilterSize,gaussfilterSize),0)
    ret,bw          = cv2.threshold(gray3, imgThresh, 255 ,cv2.THRESH_BINARY)
    #_, thresh      = cv2.threshold(gray3, th, 255,    cv2.THRESH_OTSU)
    gray3           = gray2
    gray3[bw == 0] = 0                      #all
    #------------------------
    if ( noisy ):
        out                     = bw
        out                     = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
        cv2.imshow('frame',out)
        cv2.waitKey(1000)
        if 0xFF == ord('q'):
            sys.exit()
    #---------------------
    return gray3, bw
#-------------------------------------------#
#@profile
def groupLineSegments(lines,lineMidPts,img,imgMinLineLength=2):                                         #find which two (if any) belong to a single fish
    N                       = lines.shape[0]
    gline           = []
    inds            = np.arange(lines.shape[0])
    for (i,a1) in enumerate(lines):
        qMatchedJ = False
        for (j,a2) in enumerate(lines):                                                 #
            if (j<=i):                                                                                      #its symmetric
                    continue
            if (np.sum(j==inds)==0 or np.sum(i==inds)==0):          #
                    continue
            #----
            dx      = lineMidPts[i,0] - lineMidPts[j,0]
            dy      = lineMidPts[i,1] - lineMidPts[j,1]
            D       = np.sqrt(dx**2+dy**2)
            v1      = np.array([a1[0,2]-a1[0,0],a1[0,3]-a1[0,1]])   #diff vector of line segment
            v2      = np.array([a2[0,2]-a2[0,0],a2[0,3]-a2[0,1]])   #
            v1mag   = np.sqrt(v1[0]**2+v1[1]**2)
            v2mag   = np.sqrt(v2[0]**2+v2[1]**2)
            cosang  = (v1[0]*v2[0]+v1[1]*v2[1])/v1mag/v2mag #dot prod of two vectors
            if ( (D<th_distMid) and (abs(cosang)>th_angle) ):
                if cosang>0:
                    vv = (v1+v2)/2.
                else:
                    vv = (v1-v2)/2.
                b1  = (lineMidPts[i]+lineMidPts[j])/2 - vv/2.
                b2  = (lineMidPts[i]+lineMidPts[j])/2 + vv/2.
                glinei  = [b1[0],b1[1],b2[0],b2[1] ]
                gdist   = (glinei[0]-glinei[2])**2 + (glinei[1]-glinei[3])**2
                glined  = np.array(glinei).astype(int)                          #cast as integer
                #new method; find intensities along line
                num     = int(np.hypot(glined[2]-glined[0], glined[1]-glined[3]))   # Extract the values along the line
                x, y    = np.linspace(glined[0], glined[2], num), np.linspace(glined[1], glined[3], num)
                zi      = img[ y.astype(np.int), x.astype(np.int)]           #x,y flipped in image
                zGood   = (num - np.sum(zi==0)) > th_maxZeroGroupLineImg
                if ( (gdist**0.5>imgMinLineLength) and zGood ):
                    gline.append( glinei )
                    inds = inds[inds!=i]
                    inds = inds[inds!=j]
                    qMatchedJ=True                                                                  #don't do anymore j's
            if (qMatchedJ):                                                                                 #break j loop
                break
    gline   = np.array(gline)
    return gline
#-------------------------------------------#
def findHeadPoints(glines, img):
    pt_center       = np.zeros((glines.shape[0],2))                 #centroid       pt
    pt_head         = np.zeros((glines.shape[0],2))                 #head           pt
    h_head          = np.zeros((glines.shape[0]))                   #head           intensity
    for (i,a) in enumerate(glines):
        pt_center[i,:] = [ (a[2]+a[0])/2., (a[3]+a[1])/2. ]
        #----
        a               = a.astype(int)
        indListX= np.array([a[0]-1,a[0],a[0]+1,a[0]-1,a[0],a[0]+1,a[0]-1,a[0],a[0]+1])
        indListY= np.array([a[1]+1,a[1]+1,a[1]+1,a[1],a[1],a[1],a[1]-1,a[1]-1,a[1]-1])          #test if out!!!
        innx    = (indListX>-1) * (indListX<img.shape[1])
        inny    = (indListY>-1) * (indListY<img.shape[0])
        innxy   = innx*inny
        indListX= indListX[ innxy ]
        indListY= indListY[ innxy ]                             #must be same length
        inten   = img[indListY,indListX]                        #x,y are flipped
        m1      = np.mean(inten)
        #----
        indListX= np.array([a[2]-1,a[2],a[2]+1,a[2]-1,a[2],a[2]+1,a[2]-1,a[2],a[2]+1])
        indListY= np.array([a[3]+1,a[3]+1,a[3]+1,a[3],a[3],a[3],a[3]-1,a[3]-1,a[3]-1])
        innx    = (indListX>-1) * (indListX<img.shape[1])
        inny    = (indListY>-1) * (indListY<img.shape[0])
        innxy   = innx*inny
        indListX= indListX[ innxy ]
        indListY= indListY[ innxy ]             #must be same length
        inten   = img[indListY,indListX]
        m2      = np.mean(inten)
        #----
        if      (m1>m2):
            h_head[i]       = m1
            pt_head[i]      = [a[0],a[1]]
        else:
            h_head[i]       = m2
            pt_head[i]      = [a[2],a[3]]
        #--------
    dpx             = pt_head[:,0] - pt_center[:,0]
    dpy             = pt_head[:,1] - pt_center[:,1]
    pt_ang          = np.reshape( np.arctan2(dpy,dpx), (pt_head.shape[0],1))
    h_head          = np.reshape( h_head, (pt_head.shape[0],1))
    return pt_center , pt_head,pt_ang, h_head, glines
#-------------------------------------------#
def  mergeDataij(datai,dataj):
    for i in range(dataj.shape[0]):
        dx = datai[:,0] - dataj[i,0]
        dy = datai[:,1] - dataj[i,1]
        dd = (dx**2 + dy**2)**0.5
        if (np.sum(dd<thMergeDistance)==0):
                datai = np.vstack((datai,dataj[i,:]))
    return datai
#-------------------------------------------#
#@profile
def detectUsingContours(img,imgThreshold=2,noisy=False):
    ret,bw          = cv2.threshold(img, imgThreshold, 255 ,cv2.THRESH_BINARY)
    ouut            = cv2.findContours(bw,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours        = ouut[0]
    pts             = np.ones((len(contours),6))*-1
    for (i,cnt) in enumerate(contours):
        area = cv2.contourArea(cnt)
        if len(cnt)>6:
            M = cv2.moments(cnt)
            if (M['m00']>0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                ellipse,_,angle = cv2.fitEllipse(cnt)
                pts[i,0] = cx
                pts[i,1] = cy
                pts[i,2] = angle
    pts = pts[pts[:,0]>0,:]
    #---------
    if (noisy):
        implot = plt.imshow(img)
        plt.plot(pts[:, 0], pts[:, 1], 'o', markerfacecolor=(0,1,0),markersize=7)
        plt.show()
    return pts
#-------------------------------------------#
#@profile
def detectUsingCornersLineSegments(img,gray,lsd,imgMinLineLength=2,noisyTh=False):
    lines               = lsd.detect(img)
    if ( lines is not None ):
        lineMidPts      = getMidpoints(lines)                                   #
        glines          = groupLineSegments(lines,lineMidPts,img,imgMinLineLength) #group segments, but throw out group lines not bright enough
        pt_center, pt_head, pt_angle, h_head, plines  = findHeadPoints(glines,img)              #find which point is the head...
        data            = np.hstack((pt_center,pt_angle,pt_head,h_head))        #centroid, angle, pt_head, h_intensity
    else:
        data            = np.empty((0,6))
    #---------
    if (noisyTh):
        drawn_img       = lsd.drawSegments(gray,lines)
        implot          = plt.imshow(drawn_img)
        plt.plot(pt_center[:, 0], pt_center[:, 1], 'o', markerfacecolor=(0,1,0),markersize=1)
        plt.plot(pt_head[:, 0],pt_head[:, 1], 'd', markerfacecolor=(0.8,0.2,0.2),markersize=1)
        plt.axis('off')
        plt.show()
    return data


#-------------------------------------------#
# main function
#-------------------------------------------#
def videoDetect(vname,imgThresh=20,imgMinLineLength=2,noisy=0,camData=None):
    saveName= vname[:-4]+"-detect.npz"
    print("imgThresh = %2.2f, imgMinLineLength=%2.2f, noisy = %d"%(imgThresh,imgMinLineLength,noisy))
    bg              = getBG(vname)                  #from videoFunctions                                       # load (or create) bg image from video
    if camData is not None:
        bg          = cv2.remap(bg,camData['mapx'],camData['mapy'],cv2.INTER_LINEAR)
    cap             = cv2.VideoCapture(vname)                               # load video
    Nframes         = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #--------
    tstart          = time.time()
    Ni              = 0;
    #lsd            = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)     #old version of LSD
    lsd             = cv2.ximgproc.createFastLineDetector() ### new version; fast line detector;
    #--------
    NmaxSave        = 4500          #       30*60*2.5 # save data in chunks.
    cntN            = 0             #       cnt # files saved
    saveData        = np.empty((0,6))
    t               = np.empty((0,1))
    #--------
    for i in range(Nframes):
        ret, frame  = cap.read()
        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if camData is not None:
            gray    = cv2.remap(gray,camData['mapx'],camData['mapy'],cv2.INTER_LINEAR)# undistort
        img,bw      = preProcessImage(gray,bg,imgThresh)
        datai       = detectUsingCornersLineSegments(img,gray,lsd,imgMinLineLength,noisyTh) #good to set noisyTh=0
        if (imgMinLineLength>1):     #if we tried the imgMinLineLength, merge with particles found using simple contours
            dataj   = detectUsingContours(img,imgThresh)            #
            datai   = mergeDataij(datai,dataj)
        #prepare data
        ti          = i*np.ones((datai.shape[0],1))
        Ni          += datai.shape[0]                       #only for display timing
        #--------------
        #save to tmp memory
        t           = np.vstack((t,         ti))
        saveData    = np.vstack((saveData,datai))   #centroid, angle, pt_head, h_intensity
        if ((i%NmaxSave)==0 and i>0):
            tmpName         = vname[:-4]+"-tmp-"+str(cntN)+".npz"
            np.savez(tmpName,t=t,saveData=saveData)
            saveData        = np.empty((0,6))
            t               = np.empty((0,1))
            cntN            +=1
            print(("    saved data      %d "%cntN))
        #--------------
        #only misc below, plotting
        if ( noisy ):
            out             = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            for di in datai:
                cv2.circle(out,((int)(di[0]), (int)(di[1])), 5, (200,20,20), -1)
            out     = cv2.resize(out,(0,0), fx=0.5, fy=0.5)
            cv2.imshow('image',out)
            cv2.waitKey(1000)
            if 0xFF == ord('q'):
                break
        if (i%imod ==0 and i>0):
            dt      = time.time() - tstart
            print("%04.0f Nfound_ave=%3.1f  te=%3.3f tr=%3.3f"%(i,(float)(Ni)/imod, dt, dt/(i+1)*(Nframes-i)))
            Ni = 0;
        #-------------
    dt      = time.time() - tstart
    print("Total time %3.2fmin     dt_frame= %3.3fs" % (dt/60, dt/Nframes));
    #------------------------------
    #clean up video/ffmpeg
    cap.release()
    cv2.destroyAllWindows()
    #------------------------------
    # if data was very large and we saved several files, we need to merge back
    if (cntN>0):
        tmpName         = vname[:-4]+"-tmp-"+str(cntN)+".npz"
        np.savez(tmpName,t=t,saveData=saveData)
        saveData        = np.empty((0,6))
        t               = np.empty((0,1))
        #combine data back
        for i in range(cntN+1):
            tmpName = vname[:-4]+"-tmp-"+str(i)+".npz"
            if os.path.exists(tmpName):
                with np.load(tmpName) as data:
                    ti          = data['t']
                    datai       = data['saveData']
                    t           = np.vstack((t,         ti))
                    saveData    = np.vstack((saveData,datai))  #centroid, angle, pt_head, h_intensity
                os.remove(tmpName)                          #clean up
    #------------------------------
    # prepare/save data
    saveData                = np.hstack((t,saveData))               #t, cx, cy, ang, headx, heady,headint
    np.savez_compressed(saveName, sData=saveData)
    print(saveName)
    print("finished with detect")
    print("--------")
