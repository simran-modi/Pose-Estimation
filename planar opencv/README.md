## documentation

Libraries required: 
	1. plyfile - to read ply objects
	2. numpy -  to perform array operation
	3. opencv-contrib - for SIFT and pose functions
	4. yaml - to read and write yaml files
	
Data required:
	NOTE: for textured planar objects only
	1. '.ply' file - 3D model of object to be tracked
	2. image of object to be tracked
	
Running the application:
	1. first run 'modelResgistration.py' -  click on the vertices of the object in the input image in the same order
		as listed in the .ply file.
	2. Then run 'modelDetection.py' with the object facing the camera used for real-time tracking. (The script stops
		when the object is not in view.)
