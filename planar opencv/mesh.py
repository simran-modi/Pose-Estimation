# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:48:14 2017

@author: Simran
"""

from plyfile import PlyData
import numpy as np
import cv2

list_vertex = []
list_triangle = []
num_vertex = 0
num_tri = 0

def loadMesh(ply_path):
    global list_vertex, list_triangle, num_vertex, num_tri
    list_vertex.clear()
    list_triangle.clear()
    box = PlyData.read(ply_path)
    list_vertex = [tuple([round(i,1) for i in t]) for t in box['vertex'].data]
    list_triangle = box['face'].data
    num_vertex = len(list_vertex)
    num_tri = len(list_triangle)
    
def drawObjectMesh(img, vec_2d):
    triangle = getTriangleList()
    for t in triangle:
        v1,v2,v3 = t[0]
        pts = np.array([vec_2d[v1],vec_2d[v2],vec_2d[v3]],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,[255,0,0],2)
    return img    

def getCorners():
    pts = []
    for i in range(0,num_vertex):
       pts.append(getVertex(i)) 
    return pts   
    
def getVertex(n):
    return list_vertex[n]

def getTriangleList():
    return list_triangle

def getTriangles():
    triangles = []
    for t in list_triangle:
        t = t[0]
        tri = np.array([getVertex(t[0]),getVertex(t[1]),getVertex(t[2])],np.float64)
        triangles.append(tri)
    triangles = np.array(triangles, np.float64)
    return triangles
        