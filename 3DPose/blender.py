
# Run using  blender : --background --python blender.py

import glob
import os
import numpy as np
import json
import bpy
from mathutils import Matrix,Vector

# Set directory path here
#directory_path = "C:/Users/METARVRSE/Desktop/Pose-Estimation/3DPose"
directory_path = "/home/b/gitpository/Pose-Estimation/3DPose"
os.chdir(os.path.join(directory_path,"obj_models"))

# Fetch all files and folder in obj_models
folders = os.listdir(".")

# cam - loads current camera as an object
# So cam can be moved and rotated
cam = bpy.data.objects['Camera']

# c - loads current "Camera" as a camera
# SO c can access lens properties
c = bpy.data.cameras['Camera']

# Increases clipend of c so it can render larger objects
c.clip_end = 4500

# Setting lamp (light) type to SUN
# SUN is unidirectional and at infinite distance.
# Hence only direction of SUN wrt origin matters. Not distance
lamp = bpy.data.objects['Lamp']
lamp.data.type = "SUN"

# Setting render image size to (64,64)
# Default resolution is 50% in Blender hence 0.5 *128 = 64
bpy.data.scenes['Scene'].render.resolution_x = 128
bpy.data.scenes['Scene'].render.resolution_y = 128

pose = []
object_poses = []

def look_at(point):
    '''
    Points the camera (cam) at point
    Note:
        Lamp moves along with the camera across the icosphere and point in the same direction as camera
        This is done to focus on features present in the current pose by shedding light in the same direction
    Args :
        point : Vector, to point the camera at
    '''
     global cam,lamp
     #  Fetch location of camera
     loc_camera = cam.location
     # Get directional vector
     direction = point - loc_camera
     # point the cameras '-Z' and use its 'Y' as up
     rot_quat = direction.to_track_quat('-Z', 'Y')
     # Rotate the camera and then the lamp
     cam.rotation_euler = rot_quat.to_euler()
     lamp.rotation_euler = rot_quat.to_euler()

# Appends pose (as an RT matrix) of cam to global list - pose
# RT Matrix explanation : https://www.youtube.com/watch?v=WkGSYJm2_kk&index=227&list=PLAwxTw4SYaPnbDacyrK_kB_RUkuxQBlCm
def get_3x4_RT_matrix_from_blender(cam):
    """
    Get Rotation and Translational Matrix of cam as a list
    Args:
        cam : Camera, whose RT matrix needs to be returned
    Variables :
        R_bcam2cv : Matrix representing blender's perspective of the world

    """
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
       ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))
    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    #T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam * location
    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.array(R_bcam2cv*R_world2bcam,np.float64)
    T_world2cv = np.array(R_bcam2cv*T_world2bcam,np.float64).reshape(3,1)
    # put into 3x4 matrix
    RT = np.append(R_world2cv,T_world2cv,axis=1)
    return RT.tolist()


def render_images(vertices,ID,world_matrix):
    """
    Render images of object in different poses
    Args:
        vertices : Vertices of icosphere
        ID : Number representing the ID of an object
        world_matrix : Vector to point camera at
            Ideally the center of the object which had been set to origin

    Variables :
        num : Image/Pose number
        pose : List that stores RT matrices of all poses for an object
    """
    global cam, lamp, object_poses
    num = 0
    pose = []
    # Iterating over vertices of icosphere
    for v in vertices:
        # Set camera and lamp location to vertex
        cam.location = (v.co.x,v.co.y,v.co.z)
        lamp.location = (v.co.x,v.co.y,v.co.z)
        # Look at the object
        look_at(world_matrix)
        # Get RT matrix and append to pose
        RT = get_3x4_RT_matrix_from_blender(cam)
        pose.append(RT)
        # Set filename of image and then render
        bpy.data.scenes["Scene"].render.filepath = os.path.join(directory_path,'images',str(ID),str(num)+'.jpg')
        bpy.ops.render.render(write_still=True )
        num = num + 1
    object_poses.append(pose)

def create_scene():
    """
    Iterates over all folders in the current path, checking for obj models.
    Creates an icosphere to
    Args: None
    Variables :
        folder :    Each file/folder in the directory (Each object)
        models :    List of all obj models in given folder
        obj :       Each object when loaded in blender
        max_dimension: Twice the diagonal length of object's bounding box
                    This is set as the radius of the icosphere
        ico:        Icosphere

    """
     global folders
     for folder in folders:
         if not os.path.isdir(folder):
             print (folder)
             continue
         # Changing directory to object folder if it is a directory
         os.chdir(folder)
         models = [x for x in os.listdir(".") if x.endswith(".obj")]
         print (models)
         for model in models:
             obj_name = model[0:-4]
             bpy.ops.import_scene.obj(filepath=model)
             for obj in bpy.context.selected_objects:
                 obj.name = obj_name
             obj = bpy.data.objects[obj_name]
             bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
             obj.location = (0,0,0)
             max_dimension = obj.dimensions.length * 2
             bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5,size=max_dimension,location=(0,0,0),layers=((False,)+(True,)+(False,)*18))
             ico = bpy.data.objects['Icosphere']
             vertices = ico.data.vertices
             #world_matrix = obj.matrix_world.to_translation()
             world_matrix = (0,0,0)
             render_images(vertices,obj_name,world_matrix)
             #delete the objects
             objs = bpy.data.objects
             objs.remove(objs[obj_name], True)
             objs.remove(objs['Icosphere'], True)
         # Changing directory to obj_models (previous) folder
         os.chdir("..")

create_scene()
with open(os.path.join(directory_path,'pose.json'), 'w') as outfile:
    print (outfile)
    json.dump(object_poses, outfile)
