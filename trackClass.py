#An example of a class
import numpy as np
import cv2
from kalman import KFV, KFA, KFS
     
        
class Track(object): 
	color   = (0,0,0)
	qVisible= False
	x	 	= np.zeros( (1,2)) *np.nan
	index 	= 0 	#
	#=================================================
	#kalman position
	kf 		= []
	meas 	= []
	pred 	= [] 	#predicted state
	kf_cnt 	= 0 #how many frames tracked
	kf_notFound=[]
	#=================================================
	def __init__(self):#,x0,y0,Hdif0,Hsum0):
		self.x 		= np.zeros( (1,2)) *np.nan
		self.th 	= np.nan
		self.color  = (20+235*np.random.rand(), 20+235*np.random.rand(), 20+235*np.random.rand())
		self.xhist 	= np.zeros( (0,2,) )
		self.ind 	= Track.index
		Track.index += 1							# increment counter/ belong to class, global				
		#kalman position
		self.result = []
		self.kf 	=  KFA() 						#initialize kalman filter; extended kalman, nonlinear		
		self.kf_notFound=0
		#kalman angle
		self.kf_th 	= KFS() 						#KF for theta, process_variance, estimated_measurement_variance		
		self.hist 	= np.empty((0,0)) 				#histogram Img intensity/distance; idTracker
		#area
		
    	
	def setPosition(self,x0,y0,th0):
		self.x[0,0] 	= x0
		self.x[0,1] 	= y0
		self.th 		= th0
		self.qVisible 	= True
		#----	
		if ( self.xhist.shape[0] ==0 ):
			self.xhist = self.x
		else:
			self.xhist 	= np.vstack((self.xhist,self.x))
		#kalman position	
		meas =[x0,y0]	
		if (self.kf_cnt==0): 	#initialize
			self.kf.setX(meas)	
			self.kx_th = th0
			self.kf.update(meas) 
		else:
			self.kf.update(meas) 				#update kalman filter
		self.pred 		= self.kf.pred
		#self.result.append( self.pred )
		self.kf_cnt +=1
		self.kf_notFound = 0;
		#kalman angle
		self.meas_th 	= th0
		self.kf_th.update(self.meas_th)
		self.pred_th 	= self.kf_th.pred
	#-----
	def setNotFound(self):
		#update kalman position	
		self.kf.update(self.pred) 				#update kalman filter
		self.pred 		= self.kf.pred
		self.x[0,0] 		= self.pred[0]			#update position to predictions
		self.x[0,1] 		= self.pred[1]			#update position to predictions
		self.kf_cnt = 0
		self.kf_notFound +=1
		self.qVisible = False 					#active track
		#self.xhist = np.zeros( (1,2,) )*np.nan 	#zero it out	
	#-----
	def setVisible(self, qVisible0):
		self.qVisible = qVisible0
	#-----
	def setHist(self, hist0):
		if (self.hist.shape[0]==0): 	#first time
			self.hist = hist0
		else:
			self.hist = (self.hist + hist0)/2.
	#-----
	def __exit__(self, *err):
		self.close()

