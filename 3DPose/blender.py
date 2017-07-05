
# Run using  blender : --background --python blender.py
import glob
import os
import numpy as np
import json
import bpy
from mathutils import Matrix
#directory_path = "/home/b/gitpository/Pose-Estimation/3DPose"
directory_path = ""
os.chdir(os.path.join(directory_path,"obj_models"))
folders = os.listdir(".")
cam = bpy.data.objects['Camera']
c = bpy.data.cameras['Camera']
c.clip_end = 4500
lamp = bpy.data.objects['Lamp']
lamp.data.type = "SUN"
bpy.data.scenes['Scene'].render.resolution_x = 64
bpy.data.scenes['Scene'].render.resolution_y = 64
pose = []
object_poses = []

def look_at(point):
     global cam,lamp
     loc_camera = cam.location
     direction = point - loc_camera
     # point the cameras '-Z' and use its 'Y' as up
     rot_quat = direction.to_track_quat('-Z', 'Y')
     cam.rotation_euler = rot_quat.to_euler()
     lamp.rotation_euler = rot_quat.to_euler()

def get_3x4_RT_matrix_from_blender(cam):
    global pose
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
    pose.append(RT.tolist())

def render_images(vertices,ID,world_matrix):
    global cam, lamp, pose, object_poses
    num = 0
    pose = []
    for v in vertices:
        cam.location = (v.co.x,v.co.y,v.co.z)
        lamp.location = (v.co.x,v.co.y,v.co.z)
        look_at(world_matrix)
        RT = get_3x4_RT_matrix_from_blender(cam)
        bpy.data.scenes["Scene"].render.filepath = os.path.join('images',str(ID),str(num))
        bpy.ops.render.render(write_still=True )
        num = num + 1
    object_poses.append(pose)

def create_scene():
     global folders
     for folder in folders:
         if not os.path.isdir(folder):
             print (folder)
             continue
         #print (folder)
         #print (folders)
         os.chdir(folder)
         models = glob.glob("*.obj")
         for model in models:
             obj_name = model[0:-4]
             bpy.ops.import_scene.obj(filepath=model)
             for obj in bpy.context.selected_objects:
                 obj.name = obj_name
             obj = bpy.data.objects[obj_name]
             bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
             obj.location = (0,0,0)
             max_dimension = obj.dimensions.length * 2
             bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4,size=max_dimension,location=(0,0,0),layers=((False,)+(True,)+(False,)*18))
             ico = bpy.data.objects['Icosphere']
             vertices = ico.data.vertices
             world_matrix = obj.matrix_world.to_translation()
             render_images(vertices,obj_name,world_matrix)
             #delete the objects
             objs = bpy.data.objects
             objs.remove(objs[obj_name], True)
             objs.remove(objs['Icosphere'], True)
         os.chdir("..")

create_scene()
with open(os.path.join(directory_path,'pose.json'), 'w') as outfile:
    print (outfile)
    json.dump(object_poses, outfile)
