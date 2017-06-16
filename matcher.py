# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:11:20 2017

@author: Simran
"""
"""
TO DO:
    1. Symmetry Test   
"""   
import numpy as np

ratio = 0.8
detector = None
rmatcher = None

def setDetector(d):
    global detector
    detector = d
 
def setRatio(r):
    global ratio
    ratio = r    
    
def setMatcher(m):
    global rmatcher
    rmatcher = m    
    
def computeKeypointsAndDescriptors(img):
    kp, des  = detector.detectAndCompute(img,None)
    return kp, des

def ratioTest(matches):
    good_matches = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches        

def fastRobustMatcher(frame, model_descriptors):
    frame_keypoints, frame_descriptors = computeKeypointsAndDescriptors(frame)
    if frame_descriptors is not None:
        model_descriptors = model_descriptors.astype(np.float32)
        frame_descriptors = frame_descriptors.astype(np.float32)
        matches = rmatcher.knnMatch(frame_descriptors, model_descriptors,2)
        good_matches = ratioTest(matches)
        return good_matches, frame_keypoints    
    else:
        return None, None    



