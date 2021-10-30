import numpy as np

#kalman class; 2d velocity
class KFS(object):  		#kalman filter
	def __init__(self):		
		self.x 		= np.matrix([0., 0.]).T 
		self.P 		= np.matrix(np.eye(2))*1000 	# initial uncertainty
		self.R 		= 0.1**2 						#error in measurement
		self.F = np.matrix('''  1. 0.; 
								0. 1.  ''') 
		T=0.1;  
		self.F[0,1]=T;
		self.H = np.matrix('''  1. 0. ''') 
		self.motion = np.matrix('0. 0.').T 
		self.Q = np.matrix(np.eye(2))	
		self.pred 	= 0
	#==================================
	def kalman(self,measurement ):
		'''
		Parameters:
		x: initial state
		P: initial uncertainty convariance matrix
		measurement: observed position (same shape as H*x)
		R: measurement noise (same shape as H)
		motion: external motion added to state vector x
		Q: motion noise (same shape as P)
		F: next state function: x_prime = F*x
		H: measurement function: position = H*x

		Return: the updated and predicted new values for (x, P)

		See also http://en.wikipedia.org/wiki/Kalman_filter

		This version of kalman can be applied to many different situations by
		appropriately defining F and H 
		'''
		# UPDATE x, P based on measurement m    
		# distance between measured and current position-belief
		y = np.matrix(measurement).T - self.H * self.x
		S = self.H * self.P * self.H.T + self.R  # residual convariance
		K = self.P * self.H.T * S.I    # Kalman gain
		self.x = self.x + K*y
		I = np.matrix(np.eye(self.F.shape[0])) # identity matrix
		self.P = (I - K*self.H)*self.P
		# PREDICT x, P based on motion
		self.x = self.F*self.x + self.motion
		self.P = self.F* self.P* self.F.T + self.Q
		return self.x, self.P
	#==================================
	def update(self, measurement):
		x, P = self.kalman(measurement)
		self.pred = np.squeeze(np.asarray(self.x[:1]))
		return x,P
	def setX(self, x0):
		self.x[0] = x0



#kalman class; 2d velocity
class KFV(object):  		#kalman filter
	def __init__(self):		
		self.x 		= np.matrix([0., 0., 0., 0.]).T 
		self.P 		= np.matrix(np.eye(4))*1000 	# initial uncertainty
		self.R 		= 0.1**2 						#error in measurement
		self.F = np.matrix('''  1. 0. 1 0.; 
								0. 1. 0. 1; 
								0. 0. 1. 0.; 
								0. 0. 0. 1. ''') 
		T=0.1;  
		self.F[0,2]=T;self.F[1,3]=T
		self.H = np.matrix('''  1. 0. 0. 0.; 
								0. 1. 0. 0.''') 
		self.motion = np.matrix('0. 0. 0. 0.').T 
		self.Q = np.matrix(np.eye(4))	
		self.pred = np.empty((1,2))
	#==================================
	def kalman(self, measurement):
		'''
		Parameters:
		x: initial state
		P: initial uncertainty convariance matrix
		measurement: observed position (same shape as H*x)
		R: measurement noise (same shape as H)
		motion: external motion added to state vector x
		Q: motion noise (same shape as P)
		F: next state function: x_prime = F*x
		H: measurement function: position = H*x

		Return: the updated and predicted new values for (x, P)

		See also http://en.wikipedia.org/wiki/Kalman_filter

		This version of kalman can be applied to many different situations by
		appropriately defining F and H 
		'''
		# UPDATE x, P based on measurement m    
		# distance between measured and current position-belief
		y = np.matrix(measurement).T - self.H * self.x
		S = self.H * self.P * self.H.T + self.R  # residual convariance
		K = self.P * self.H.T * S.I    # Kalman gain
		self.x = self.x + K*y
		I = np.matrix(np.eye(self.F.shape[0])) # identity matrix
		self.P = (I - K*self.H)*self.P
		# PREDICT x, P based on motion
		self.x = self.F*self.x + self.motion
		self.P = self.F* self.P* self.F.T + self.Q
		return self.x, self.P
		
	#==================================
	def setX(self, x0):
		self.x[0] = x0[0]
		self.x[1] = x0[1]
	def update(self, measurement):
		x, P = self.kalman(measurement)
		self.pred = np.squeeze(np.asarray(self.x[:2]))
		return x,P
		
		
#kalman class
class KFA(object):  		#kalman filter
	def __init__(self):		
		T = 0.1
		self.x 		= np.matrix([0., 0., 0., 0., 0., 0.]).T 
		self.P 		= np.matrix(np.eye(6))*1000 	# initial uncertainty
		self.R 		= 0.1**2 						#error in measurement
		self.F = np.matrix([[1.,0.,T, 0.,T*T/2.,0.],
							[0.,1.,0.,T, 0.,    T*T/2.],
							[0.,0.,1.,0.,T, 	0.],
							[0.,0.,0.,1.,0.,	T 	],
							[0.,0.,0.,0.,1.,	0.],
							[0.,0.,0.,0.,0.,	1.]])
		

		self.H = np.matrix([[1.,0.,0, 0.,0.,0.],							
							[0.,1.,0.,0.,0.,0.]])
		
		self.motion = np.matrix([0., 0., 0., 0., 0., 0.]).T  	#state
		self.Q = np.matrix(np.eye(6))	
		self.pred = np.empty((1,2))
	#==================================
	def kalman(self, measurement ):
		'''
		Parameters:
		x: initial state
		P: initial uncertainty convariance matrix
		measurement: observed position (same shape as H*x)
		R: measurement noise (same shape as H)
		motion: external motion added to state vector x
		Q: motion noise (same shape as P)
		F: next state function: x_prime = F*x
		H: measurement function: position = H*x

		Return: the updated and predicted new values for (x, P)

		See also http://en.wikipedia.org/wiki/Kalman_filter

		This version of kalman can be applied to many different situations by
		appropriately defining F and H 
		'''
		# UPDATE x, P based on measurement m    
		# distance between measured and current position-belief
		y = np.matrix(measurement).T - self.H * self.x
		S = self.H * self.P * self.H.T + self.R  # residual convariance
		K = self.P * self.H.T * S.I    # Kalman gain
		self.x = self.x + K*y
		I = np.matrix(np.eye(self.F.shape[0])) # identity matrix
		self.P = (I - K*self.H)*self.P
		# PREDICT x, P based on motion
		self.x = self.F*self.x + self.motion
		self.P = self.F* self.P* self.F.T + self.Q
		return self.x, self.P
	def setX(self, x0):
		self.x[0] = x0[0]
		self.x[1] = x0[1]
	#==================================
	def update(self, measurement):
		self.x, self.P = self.kalman(measurement)
		self.pred = np.squeeze(np.asarray(self.x[:2]))
		return self.x, self.P
	

