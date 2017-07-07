DOCUMENTATION

GOAL: To estimate the 3D pose of an object in real time.

To identify the 3D pose of an object it is sufficient to find the projection matrix that maps the transformation from
3D Euclidean space to 2D image. It is also known as the extrinsic camera parameters.

													Xc = R(Xw) + T
				
											Xc -	The camera view coordinates
											Xw -	The world view coordinates
											R  -	3x3 rotational matrix
											T  -	3x1 translational vector

Concatenating the R and T i.e. [R|T] results in a 3x4 projection matrix.

APPROACH 1:
	
	The initial approach follows from link: http://docs.opencv.org/trunk/dc/d2c/tutorial_real_time_pose.html
	
	Implementation:
	
		MODEL REGISTRATION
		1. Create a 3D model (here .ply file) of the object you intend to track - gives 3D point of the corners
		2. Take a image of the object such that maximum number of corners of the object is visible
		3. Click on the corners of the object in the image - gives 3D points of the corners
		4. Given the 2D and 3D points of the vertices and intrinsic camera parameters, PnP+RANSAC estimates
			the projection matrix P
		5. Find the SIFT features of the object from the image - gives 2D feature points
		6. Using inverse of P and the 2D feature points the 3D feature points are identified and saved along with
			their descriptors
			
		MODEL DETECTION
		1. For real-time tracking, identify the 2D SIFT features in every frame and match it's descriptors with the
			saved feature descriptors to find the corresponding 3D feature points
		2. Estimate the projection matrix P using pnp+RANSAC given the 2D and 3D feature points
		3. Use P to draw the pose as a mesh around the object

	Cons:
		1. It works only for planar textured objects
		2. SIFT is proprietary. ORB as an alternative has low accuracy
		3. Computing SIFT features is slow
		4. It fails for unseen face(That was not visible in the registration image) of the object.
	
	Suggestions:
		1. Implement Kalman filter for smoother pose detection
		2. Map features detected during registration to the face it was detected in. Repeat for all faces to 
			address unseen face issue.
		
		
			
	
									
