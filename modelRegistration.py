# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:58:07 2017

@author: Simran
"""
"""
NOTE:
    1.
TO DO:
    1. automatic camera parameters generation?
    2. automatic corner detection?
"""

import cv2
import mesh
import projection as proj
import matcher
import model

NUM_CORNERS = 8
n_corners = 0

points_2d = []
points_3d = []

img_path = "Data/resize.jpg"
ply_path = "Data/box.ply"
yaml_path = "Data/cookies_ORB.yml"

# Intrinsic camera parameters: UVC WEBCAM
f = 45 #focal length in mm
sx = 22.3
sy = 14.9
width = 2592
height = 1944
cam_params = [width*f/sx,  # fx
             height*f/sy,  # fy
             width/2,      # cx
             height/2]    # cy

#draw points on image
def drawMeshAndPoints(vec_2d, img):
    img = model.draw2DKeypoints(img)
    img = mesh.drawObjectMesh(img, vec_2d)
    cv2.imshow("Model",img)    

#Mouse callback function
def on_mouse_click (event, x, y, flags, img_clone):
    global n_corners
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img_clone,(x,y),5,[255,0,0],3)
        cv2.imshow("box", img_clone)
        points_2d.append([x,y])
        points_3d.append(list(mesh.getVertex(n_corners)))
        n_corners = n_corners + 1
        if n_corners == NUM_CORNERS:
            cv2.destroyAllWindows()
       
#Ask user to select the corners   
def selectCorners(img_path):
    cv2.namedWindow("box")
    img = cv2.imread(img_path)
    cv2.imshow("box",img)
    cv2.setMouseCallback("box",on_mouse_click, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#Register the 3D Model - compute 3D points of the features
def registerModel(imgp):
    model.clearLists
    img = cv2.imread(imgp)
#    orb = cv2.ORB_create(1000)
#    matcher.setDetector(orb)
    sift = cv2.xfeatures2d.SIFT_create()
    matcher.setDetector(sift)
#    surf = cv2.xfeatures2d.SURF_create()
#    matcher.setDetector(surf)
    keypoints, descriptors = matcher.computeKeypointsAndDescriptors(img)
    #register keypoints on the surface as inliers and others as outliers
    for i in range(0,len(keypoints)):
        pt_2D = keypoints[i].pt
        on_surface, pt_3D = proj.backProject2DPoint(pt_2D)
        if on_surface:
            model.addCorrespondence(pt_2D,pt_3D)
            model.addFeatures(keypoints[i],descriptors[i])
        else:
            model.addOutlier(pt_2D) 
    model.save(yaml_path)        

def main():
    mesh.loadMesh(ply_path)
    selectCorners(img_path)
    proj.initializeCameraMatrix(cam_params)
    proj.estimateCameraPose(points_3d, points_2d)
    mesh_2d_pts = proj.backProject3DPoint(points_3d)
    registerModel(img_path)
    img = cv2.imread(img_path)
    drawMeshAndPoints(mesh_2d_pts,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
main()    