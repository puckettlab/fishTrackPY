import numpy as np
import cv2
#=================
def image2world(uv,K,R,t,dist=None):
	if (dist is None):
		cx 		= K[0,2]
		cy 		= K[1,2]
		fx 		= K[0,0]
		fy 		= K[1,1]
		uv[:,0] = (uv[:,0]-cx)/fx
		uv[:,1] = (uv[:,1]-cy)/fy
	else:
		uv 		= undistort_point(uv, K, dist)
	z 		= np.ones(uv[:,0].shape)
	xyz 	= np.vstack((uv.T,z)) 		#homogeneous camera coordinate
	dxyz 	= np.dot(R.T,xyz) 			#image to world
	Txyz 	= np.dot(R.T,-t) 			#camera position
	zz 		= -Txyz[2]/dxyz[2,:] 		#https://stackoverflow.com/questions/14514357/converting-a-2d-image-point-to-a-3d-world-point?rq=1
	xx 		= zz*dxyz[0,:]+Txyz[0]
	yy 		= zz*dxyz[1,:]+Txyz[1]
	xyz[0,:]=xx
	xyz[1,:]=yy
	xyz[2,:]=0
	#
	return xyz.T

#=================
def world2image(xyz,K,R,t,dist=None):
	#works!!!!, difference due to distortion
	#similar to projectPoints
	if (xyz.shape[0]==1):
		xyz = xyz.reshape( (xyz.shape[1],3) )
	xyz1 	= np.hstack((xyz,np.ones((xyz.shape[0],1))))
	P 		= np.dot(K, np.hstack((R,t))) 			#P*[R|t]
	uv 		= np.dot(P,xyz1.T).T 						#P*[R|t]*[x,y,z,1].T
	xycam   = np.vstack((uv[:,0]/uv[:,2],uv[:,1]/uv[:,2])).T	#rescale to homogeneous
	if (dist is None):
		return xycam
	else:
		return distortBackPoints(xycam, K, dist)
#=================
def undistort_point(uv, K, dist_coeffs):
	# see link for reference on the equation
	# http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
	# make more efficient?? r -> r2
	k1,k2,p1,p2, k3 = dist_coeffs[0]
	cx = K[0,2] # cx
	cy = K[1,2] # cy
	fx = K[0,0]
	fy = K[1,1]
	x = (uv[:,0] - cx)/fx 			# homogeneous coords
	y = (uv[:,1] - cy)/fy 			# homogeneous coords
	r = np.sqrt(x**2 + y**2)
	u_undistort = (x * (1+ (k1*r**2) + (k2*r**4) + (k3*r**6))) + 2*p1*x*y + p2*(r**2 + 2*x**2)
	v_undistort = (y * (1+ (k1*r**2) + (k2*r**4) + (k3*r**6))) + 2*p2*y*x + p1*(r**2 + 2*y**2)
	uv 	= np.vstack((u_undistort,v_undistort)).T
	return uv
#=================
def distortBackPoints(xy, K, dist):
	fx = K[0,0]
	fy = K[1,1]
	cx = K[0,2]
	cy = K[1,2]
	k1 = dist[0][0] * -1
	k2 = dist[0][1] * -1
	k3 = dist[0][4] * -1
	p1 = dist[0][2] * -1
	p2 = dist[0][3] * -1
	x = (xy[:,0] - cx) / fx
	y = (xy[:,1] - cy) / fy

	r2 = x*x + y*y

	xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
	yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

	xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
	yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

	xDistort = xDistort * fx + cx;
	yDistort = yDistort * fy + cy;
	xyDistort = np.vstack((xDistort, yDistort)).T
	return xyDistort



#
#=================
def drawCircles(img,xy,col=(0,250,20)):
	xy 	= xy.reshape((xy.shape[0],2))
	r  	= 5
	for i in range(xy.shape[0]):
		if (np.sum( np.isnan(xy[i]))==0):
			cv2.circle(img,((int)(xy[i,0]), (int)(xy[i,1])), r, col, -1)
	return img

#=================
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img1
        lines - corresponding epilines '''
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = list(map(int, [0, -r[2]/r[1] ]))
        x1,y1 = list(map(int, [c, -(r[2]+r[0]*c)/r[1] ]))
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1), 10, color, -1)
        cv2.circle(img2,tuple(pt2), 10,color,-1)
    return img1,img2
