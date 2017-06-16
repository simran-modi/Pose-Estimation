# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:37:48 2017

@author: Simran
"""

import cv2
import yaml
import numpy as np

inliers_2D = []
outliers_2D = []
inliers_3D = []
list_keypoints = []
list_descriptors = []

def clearLists():
    inliers_2D.clear()
    outliers_2D.clear()
    inliers_3D.clear()
    list_keypoints.clear()
    list_descriptors.clear()
    
def get_points2D_out():
    return np.array(outliers_2D)

def get_points2D_in():
    return np.array(inliers_2D)

def get_points3D():
    return np.array(inliers_3D)

def get_keypoints():
    return np.array(list_keypoints)

def get_descriptors():
    return np.array(list_descriptors)

def addCorrespondence(pt_2D, pt_3D):
    inliers_2D.append(pt_2D)
    inliers_3D.append(pt_3D)

def addOutlier(pt_2D):
    outliers_2D.append(pt_2D)

def addFeatures(kp,des):
    list_keypoints.append(kp)
    list_descriptors.append(des)    
    
def draw2DKeypoints(img):
    for pt in inliers_2D:
        cv2.circle(img,(int(pt[0]),int(pt[1])),3,[0,255,0],2)
    for pt in outliers_2D:    
        cv2.circle(img,(int(pt[0]),int(pt[1])),3,[0,0,255],2) 
    return img

# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    if mapping["cols"] > 1:
        mat.resize(mapping["rows"], mapping["cols"])
    else:
        mat.resize(mapping["rows"], )
    return mat

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if mat.ndim > 1:
        mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    else:
        mapping = {'rows': mat.shape[0], 'cols': 1, 'dt': 'd', 'data': mat.tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)

def initYamlParser():
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
    yaml.add_representer(np.ndarray, opencv_matrix_representer)

def save(write_path):
    initYamlParser()
    data = {"points_3d": get_points3D(),
            "points_2d": get_points2D_in(),
            "descriptors": get_descriptors()}
    with open(write_path, 'w') as outfile:
        yaml.dump(data, outfile)
    outfile.close()    
    
def load(read_path):
    global inliers_3D, list_descriptors
    with open(read_path, 'r') as infile:
        data = yaml.load(infile)   
    inliers_3D = data["points_3d"]
    list_descriptors = data["descriptors"]