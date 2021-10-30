import os
import sys
import numpy as np
import scipy
#from scipy.interpolate import splprep, splev,interp1d
#-------------------------------------------#
#linking parameters
delta		= 1 			#missing frames between links; connect upto 1s gap in delta seconds
tau     	= 0.5 			#splice trajectories even if overlap of this size in tau seconds
minTrackLength = 10 		#need at least this many frames in a track
CmaxSplice 	= 30 			#80 ok?
#-------------------------------------------#
def distance(x1,x2):
	return np.sqrt( np.sum( (x1-x2)**2) )
#-------------------------------------------#
def getOverlapCost(xi,xj,ti,tj,vi,vj):
    tie 	= ti[-1]
    tjb 	= tj[0]
    vie     = vi[-1,:] 				        #final   velocity of i'th
    vjb 	= vj[0,:] 					    #initial velocity of j'th
    dt      = (tjb-tie)/2.
    xip     = xi[-1,:]- vie * (dt)   	    #end of xi, 		project forward to midpoint
    xjp     = xj[0,:] + vjb * (dt) 	        #beginning of xj,   project back to midpoint
    d       = distance(xip,xjp)
    return d
#-------------------------------------------#
def getProjectedCost(xi,xj,ti,tj,vi,vj):
    tie 	= ti[-1]
    tjb 	= tj[0]
    vie     = vi[-1,:] 				        #final   velocity of i'th
    vjb 	= vj[0,:] 					    #initial velocity of j'th
    dt      = (tjb-tie)/2.
    xip     = xi[-1,:]+ vie * (dt)   	    #end of xi, 		project forward to midpoint
    xjp     = xj[0,:] - vjb * (dt) 	        #beginning of xj,   project back to midpoint
    d       = distance(xip,xjp)
    return d
#-------------------------------------------#    
def getInterpolate(ti,tj,xi,xj,tbetween):
    xij     = np.vstack((xi,xj))
    tij     = np.hstack((ti,tj))
    fx      = scipy.interpolate.interp1d(tij, xij[:,0])
    fy      = scipy.interpolate.interp1d(tij, xij[:,1])
    fz      = scipy.interpolate.interp1d(tij, xij[:,2])
    fxnew   = fx(tbetween)
    fynew   = fy(tbetween)
    fznew   = fz(tbetween)
    xbetween= np.vstack((fxnew,fynew,fznew)).T
    return xbetween
#-------------------------------------------#
def linkTracks(t,ind,x,v,a):
    ut 		= np.unique(t)
    uind 	= np.unique(ind)
    #Find which tracks are possible matches?????
    Cinf    = 10000.
    C  		= np.ones((uind.shape[0],uind.shape[0]))*Cinf 	#kinematic cost
    Ct 		= np.zeros((uind.shape[0],uind.shape[0]))       #type
    for (i,ui) in enumerate(uind):
        for (j,uj) in enumerate(uind):
            if (ui!=uj):
                indi    = ind==ui
                ti 		= t[indi]
                xi      = x[indi]
                vi      = v[indi]
                indj    = ind==uj
                tj 		= t[indj]
                xj      = x[indj]
                vj      = v[indj]
                #---
                tie 	= ti[-1] 				#end of i
                tjb 	= tj[0] 				#begininning of j
                dt      = tjb - tie
                if   ( (dt>0) and (dt < delta) and Ct[j,i]==0): 	   #j occurs at most delta frames after i
                    cost 	= getProjectedCost(xi,xj,ti,tj,vi,vj)      #projected cost, constant vel model
                    C[i,j]  = cost
                    Ct[i,j] = 1
                elif ( (dt<0) and (np.abs(dt) < tau) and Ct[j,i]==0):  #tracks overlap by less than tau, ct makes sure that we are not double counting here
                    cost 	= getOverlapCost(xi,xj,ti,tj,vi,vj)		   #overlap cost
                    C[i,j]  = cost
                    Ct[i,j] = 2
        if (i>20):
            break
    #-----
    print ("finished with cost matrix")
    print ("performing linear assignment")
    row, col = scipy.optimize.linear_sum_assignment(C) 			#pos, column is the matched id; fit with hungarian algorithm
    print ("finished with linear assignment")
    indNew = ind.copy()
    x_interp=[]
    v_interp=[]
    a_interp=[]
    t_interp=[]
    ind_interp=[]
    for i in range(row.shape[0]):
        Ci      = C[row[i],col[i]]
        Ctype   = Ct[row[i],col[i]]
        if (Ci<CmaxSplice):
            ui      = uind[row[i]]
            uj      = uind[col[i]]
            indi    = ind==ui
            indj    = ind==uj
            ti 		= t[indi]
            tj 		= t[indj]
            tie 	= ti[-1] 			        #end of i
            tjb 	= tj[0] 				    #begininning of j
            dt      = tjb - tie
            innt    = (ut>tie) * (ut<tjb)
            if (Ctype==1):                      #projected, but tjb = tie+dt; no need to interpolate
                if np.sum(innt)==0:
                    indNew[ui==ind] = ui
                    indNew[uj==ind] = ui
                    Ct[col[i],row[i]]=Cinf      #make sure we don't repeat this for pair (j,i); very short track
                else:                           #need to interpolate
                    tbetween= ut[innt]
                    xbetween= getInterpolate(ti,tj,x[indi],x[indj],tbetween)
                    vbetween= getInterpolate(ti,tj,v[indi],v[indj],tbetween)
                    abetween= getInterpolate(ti,tj,a[indi],a[indj],tbetween)
                    indNew[ui==ind] = ui
                    indNew[uj==ind] = ui
                    x_interp.append(xbetween)
                    v_interp.append(vbetween)
                    a_interp.append(abetween)
                    t_interp.append(tbetween)
                    ind_interp.append(ui)
                    Ct[col[i],row[i]]=Cinf      #make sure we don't repeat this for pair (j,i); very short track
            elif (Ctype==2): #dt<0, overlap, just average, or interpolate from endpoints?
                print('ahh')    #also dont want to merge trajectories that are close (overlap), but different fish.  another C_threshold??


            #but now fix interpolation or overlap
    #-----
    #finally, merge interp_tmp
    #-----
    print ("Inds old shape=%d, new=%d"%(uind.shape[0],np.unique(indNew).shape[0]))
    sys.exit()
    return t,indNew,x
#-------------------------------------------#
def videoTrackSplice(vname,noisy=0,camData=None):
    loadName    = vname[:-4]+"-vtracks.npz"
    saveName    = vname[:-4]+"-vtracks-splice.npz"             #tracks in World coordinates
    if ( not os.path.isfile(loadName) ):
        print ("Looking for file: %s"%loadName)
        print ("No detect file found")
        sys.exit()
    #----
    data        = np.load(loadName);
    x           = data['x'] ##xy in pixel coords
    v           = data['v'] ##xy in pixel coords
    a           = data['a'] ##xy in pixel coords
    t           = data['t']
    ind         = data['ind']
    uind        = np.unique(ind)
    #----
    dataX       = np.zeros((0,3))       #position
    dataV       = np.zeros((0,3))       #velocity
    dataA       = np.zeros((0,3))       #acceleration
    dataT       = np.zeros((0,))        #time
    dataI       = np.zeros((0,))        #index
    #----
    t,ind,x     = linkTracks(t,ind,x,v,a) #
    #------------------------------
    print(dataX.shape)
    #np.savez_compressed(saveName,x=dataX,v=dataV,a=dataA,t=dataT,ind=dataI)
    print("finished converting to world coordinates")
