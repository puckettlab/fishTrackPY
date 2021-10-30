#!/usr/bin/env python
#
# python3 cameraCalibration.py calibImages/
#
'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]
default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''
# Python 2/3 compatibility
import numpy as np
import cv2
import yaml 					#save data to human readable format
import os
from cameraFunctions import *
import sys
import getopt
from glob import glob
#=================
gamma 			= 1.9 			#increase gamma
#=================
rows 			= 27
cols 			= 22
filetype 	    = 'jpg'
qSaveFit        = 1
#=================
square_size_Default = 25.4 	        #calibration grid spacing in mm
px_sensor 		= 5.5e-6 			#sensor size, retrieve from camera specs in m
#=================
noisy_Default 	= 1
# termination criteria
criteria        = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#=================
def processImage(fn,noisy):
    print('processing %s... ' % fn)
    img = cv2.imread(fn)
    if img is None:
        print("Failed to load", fn)
        return None
    gray 	= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe 	= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray 	= clahe.apply(gray)
    gray 	= cv2.multiply(gray,np.array([gamma]))
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=	cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
    if found:
        corners2    = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        if noisy:
            img 	= cv2.drawChessboardCorners(img, (rows, cols), corners2, found)
            img2 	= cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            cv2.imshow('img', img2)
            cv2.waitKey(50)
            if qSaveFit:
                cv2.imwrite(fn[:-4]+'-cb.jp2',img2)
        print('Found: %s... OK' % fn)
        return (corners2.reshape(-1, 2), pattern_points)
    else:
        print('Not found 	      	%s' % fn)
        return None
#=================
if __name__ == '__main__':
    #=================
    #handle input arguments
    args, fname0 = getopt.getopt(sys.argv[1:], '', ['square_size=', 'threads=','noisy='])
    args        = dict(args)
    args.setdefault('--square_size', square_size_Default)
    args.setdefault('--threads', 4)
    args.setdefault('--noisy', noisy_Default)
    img_names  	= sorted(glob(fname0[0]+'/*'+filetype))
    fDir, fName = os.path.split(img_names[0])
    square_size = float(args.get('--square_size'))
    threads_num = int(args.get('--threads'))
    noisy 		= int(args.get('--noisy'))
    if noisy:
    	threads_num=1
    #=================
    print('here')
    cameraFname 		= fDir+'/cameraCalib.npz'
    cameraFnameYaml 	= fDir+'/cameraCalib.yaml'
    pattern_size        = (rows, cols)
    pattern_points      = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points      *= square_size 					#convert to mm
    obj_points          = []
    img_points          = []
    h, w                = cv2.imread(img_names[0], 0).shape[:2]  	#
	#======================
    threads_num=1
    if threads_num <= 1:
        chessboards = [processImage(fn,noisy) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names) 		#returns corners w/ subpixel resolution
    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
    #======================
    # calculate camera distortion
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    #======================
    # improve camera calibration
    #======================
    _, rvec_new, tvec_new   = cv2.solvePnP(pattern_points, img_points[0], K, dist)
    K_new, roi              = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    mapx,mapy 		        = cv2.initUndistortRectifyMap(K,dist,None,K_new,(w,h),5)
	#======================
    tot_error   = 0
    for i in range(len(obj_points)):
    	imgpoints2, _  = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    	error          = cv2.norm(img_points[i],imgpoints2[:,0,:], cv2.NORM_L2)/len(imgpoints2) #need to slice imgpoints2
    	tot_error      += error
    FL          = K[0][0]*px_sensor*1000
    R ,_		= cv2.Rodrigues(rvecs[0])
    R_new ,_	= cv2.Rodrigues(rvec_new)
    #======================
    #save yaml file
    np.savez(cameraFname, K=K, R=R,tvec=tvecs[0],dist=dist,mapx=mapx,mapy=mapy,roi=roi,corners=img_points[0],K_new=K_new,R_new=R_new,tvec_new=tvec_new)
    #data = {'K': np.asarray(K).tolist(), 'R': np.asarray(R).tolist(),'tvec': np.asarray(tvecs[0]).tolist(),'dist': np.asarray(dist).tolist(),'K_new': np.asarray(K_new).tolist(), 'R_new': np.asarray(R_new).tolist(),'tvec_new': np.asarray(tvec_new).tolist()}
    #with open(cameraFnameYaml, "w") as f:
    #	yaml.dump(data, f)
    #======================
    data 			=np.load(cameraFname)
    K 				=data['K']
    dist 			=data['dist']
    R 				=data['R']
    tvec			=data['tvec']
    tvec_new 		=data['tvec_new']
    K_new 			=data['K_new']
    R_new 			=data['R_new']
    #======================
    print("camera matrixN:          \n", K_new)
    print("camera matrix:           \n", K)
    print("distortion coefficients: \n", dist.ravel())
    print("R:                       \n", R.ravel())
    print("T:                       \n", tvec.ravel())
    print("RMS:                     ", rms)
    print("total error:             ", tot_error/len(obj_points))
    print("camera focal length:     ", FL)
    cv2.destroyAllWindows()
    #======================
