# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:59:35 2017

@author: Simran
"""

"""
NOTE:
    1. pictures are not undistorted - try that maybe? 
"""

import cv2
import numpy as np
from numpy.linalg import inv,norm
import mesh

A = np.zeros((3,3),np.float64)
P = np.zeros((4,3),np.float64)
r_matrix = np.zeros((3,3),np.float64)
t_matrix = np.zeros((3,1),np.float64)

#Initialize the intrinsic camera parameters
def initializeCameraMatrix(cam):
    global A
    A[0,0] = cam[0]
    A[1,1] = cam[1]
    A[0,2] = cam[2]
    A[1,2] = cam[3]
    A[2,2] = 1
     
def initializePoseMatrix(r_matrix,t_matrix):
    global P
    P = np.append(r_matrix,t_matrix,1)    

#estimate camera pose
def estimateCameraPose(points_3d,points_2d):
    global P, r_matrix, t_matrix
    #distortion not taken into account
    ret, rvec, tvec = cv2.solvePnP(np.array(points_3d,np.float64),
                                   np.array(points_2d,np.float64),
                                   A,None,useExtrinsicGuess=False,
                                   flags = cv2.SOLVEPNP_ITERATIVE)
    r_matrix, jac = cv2.Rodrigues(rvec)
    t_matrix = tvec
    initializePoseMatrix(r_matrix,t_matrix)
    
def estimatePoseRANSAC(points_3d,points_2d,reprojectionError,iterationsCount,confidence):
    global A, r_matrix, t_matrix
    pts_2d = np.array(points_2d,np.float64).reshape((-1,1,2)) #if you don't reshape, error
    pts_3d = np.array(points_3d,np.float64)
    dist = np.zeros((4,1),np.float64)
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d,
                                                  A,dist,useExtrinsicGuess=False,
                                                  reprojectionError = reprojectionError,
                                                  iterationsCount = iterationsCount,
                                                  confidence = confidence,
                                                  flags = cv2.SOLVEPNP_ITERATIVE)
    r_matrix, jac = cv2.Rodrigues(rvec)
    t_matrix = tvec
    initializePoseMatrix(r_matrix,t_matrix)
    return inliers
    
def backProject3DPoint(pts_3D):
    list_vec_2d = []
    for pt3 in pts_3D:
        vec_3d = np.append(pt3,1)
        vec_2d = A.dot(P.dot(vec_3d))
        list_vec_2d.append(tuple((vec_2d / vec_2d[2]).astype(int)[:2]))
    return list_vec_2d

def backProject2DPoint(pt_2D):
    LAMBDA = 8
    vec_2d = np.array([[pt_2D[0]*LAMBDA],[pt_2D[1]*LAMBDA],[LAMBDA]],np.float64)
    #Point in camera coordinates
    X_c = (inv(A)).dot(vec_2d) # 3x1
    #Point in world coordinates
    X_w = (inv(r_matrix)).dot(X_c - t_matrix) #3x1
    #Center of projection - Ray origin
    C_op = (inv(r_matrix)*-1).dot(t_matrix) # 3x1    
    #Ray direction vector
    ray = X_w - C_op # 3x1
    ray = ray / cv2.norm(ray) # 3x1
    #ray-triangle intersection Moller-Trumbore algorithm
    triangles = mesh.getTriangles()
    intersections = intersect_MollerTrumbore(C_op,ray,triangles)
    if len(intersections) == 0:
        return False, None
    else:
        #find nearest intersection point
        dist = []
        for pt in intersections:
            dist.append(norm(pt-C_op))      
        return True, intersections[dist.index(min(dist))]    
            


def intersect_MollerTrumbore(C_op, ray, triangles):
    intersections = []
    EPSILON = 0.000001
    O = C_op[:,0] # Ray origin
    D = ray[:,0] # Ray direction
    for tri in triangles:
        V1 = tri[0]  #Triangle vertices
        V2 = tri[1]
        V3 = tri[2]
        #Find vectors for two edges sharing V1
        e1 = V2 - V1
        e2 = V3 - V1
        # Begin calculation determinant - also used to calculate U parameter
        Pv = np.cross(D,e2)
        # If determinant is near zero, ray lie in plane of triangle
        det = e1.dot(Pv)
        #NOT CULLING
        if det > -EPSILON and det < EPSILON:
            continue
        inv_det = 1.0 / det             
        #calculate distance from V1 to ray origin
        Tv = O - V1
        #Calculate u parameter and test bound
        u = Tv.dot(Pv) * inv_det     
        #The intersection lies outside of the triangle
        if u < 0.0 or u > 1.0:
            continue
        #Prepare to test v parameter
        Qv = np.cross(Tv,e1)
        #Calculate V parameter and test bound
        v = D.dot(Qv) * inv_det
        #The intersection lies outside of the triangle
        if v < 0.0 or u + v  > 1.0:
            continue
        t = e2.dot(Qv) * inv_det;  
        if t > EPSILON:
            intersections.append((O + (t*D)))
    return intersections