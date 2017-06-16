# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:14:56 2017

@author: Simran
"""

"""
NOTE:
    1. runs better with SIFT features than ORB but is slower.
    2. SURF is not as good as SIFT, seems equally slow. Adjust parameters maybe?
    
TO DO:
    1. Implement kalman filter
    2. Try getting features from multiple algorithms
"""    

import cv2
import mesh
import model
import matcher
import projection as proj



list_points3d_model = []
list_points2d_scene = []
corners_3d =[]

video_read_path = "Data/box.mp4"
yml_read_path = "Data/cookies_ORB.yml"
ply_read_path = "Data/box.ply"

#Intrinsic camera parameters: UVC WEBCAM
f = 55
sx = 22.3
sy = 14.9             
width = 640 
height = 480        
cam_params = [ width*f/sx,  # fx
              height*f/sy,  # fy
              width/2,      # cx
              height/2]     # cy

# Robust Matcher parameters
numKeyPoints = 2000   # number of detected keypoints
ratioTest = 0.70      # ratio test

# RANSAC parameters
iterationsCount = 500      # number of Ransac iterations.
reprojectionError = 2.0    # maximum allowed distance to consider it an inlier.
confidence = 0.95          # ransac successful confidence.


def initialize():
    global corners_3d
    proj.initializeCameraMatrix(cam_params)
    mesh.loadMesh(ply_read_path)
    corners_3d = mesh.getCorners()
    model.load(yml_read_path)
#    FLANN_INDEX_LSH = 0
#    orb = cv2.ORB_create(numKeyPoints)
#    matcher.setDetector(orb)
#    search_params = dict(checks = 50)
#    index_params = dict(algorithm = FLANN_INDEX_LSH,
#                        table_number = 6,
#                        key_size = 12,
#                        multi_probe_level = 1) 
#    surf = cv2.xfeatures2d.SURF_create()
#    matcher.setDetector(surf)
    sift = cv2.xfeatures2d.SIFT_create()
    matcher.setDetector(sift)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matcher.setMatcher(flann)
    matcher.setRatio(ratioTest)
    
def detectModel(video_path):
    model_3d = model.get_points3D()
    model_descriptors = model.get_descriptors()
    cap = cv2.VideoCapture(video_path)
    ret = True
    while(ret):
        ret, frame = cap.read()
        #Step 1: Robust matching between model descriptors and scene descriptors
        good_matches, scene_keypoints = matcher.fastRobustMatcher(frame,model_descriptors)
        if good_matches is None:
            continue
        #Step 2: Find out the 2D/3D correspondences
        list_points3d_model.clear()
        list_points2d_scene.clear()
        for m in good_matches:
            list_points3d_model.append(model_3d[m.trainIdx])
            list_points2d_scene.append(scene_keypoints[m.queryIdx].pt)
        #Step 3: Estimate the pose using RANSAC approach
        inliers_idx = proj.estimatePoseRANSAC(list_points3d_model,
                                          list_points2d_scene,
                                          reprojectionError,
                                          iterationsCount,
                                          confidence)
        #Step 4: Draw the inliers and Mesh 
        if inliers_idx is not None:
            for idx in inliers_idx:
                pt = (int(list_points2d_scene[idx[0]][0]),int(list_points2d_scene[idx[0]][1]))
                cv2.circle(frame,pt,3,[0,255,0],2)   
            vec_2d = proj.backProject3DPoint(corners_3d)
            frame = mesh.drawObjectMesh(frame,vec_2d)    
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
        
def main():
    initialize()
    detectModel(video_read_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

main()    