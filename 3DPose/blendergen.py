import glob
import os
import numpy as np
import platform

if os.name == "posix":
    directory = "/home/b/Downloads/duck/"
elif os.name == "nt":
    directory = "C:/Users/Simran/Desktop/Code/models/duck/"

os.chdir(directory)
models = glob.glob("*.ply")
cam = bpy.data.objects['Camera']

world = bpy.context.scene.world
world.horizon_color = (255, 255, 255)

model = models[-1]
bpy.ops.import_mesh.ply(filepath=model)
obj_name = model[0:-4]
obj = bpy.data.objects[obj_name]
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
obj.location = (0,0,0)
if obj.dimensions > Vector((35,35,35)):
   obj.scale = (0.03,0.03,0.03)

step_count = 20
obj.rotation_euler[0] = radians(180)
obj.rotation_euler[1] = radians(0)
bpy.context.space_data.viewport_shade = 'TEXTURED'
num = 0
for step in range(-1*step_count, step_count):
    obj.rotation_euler[2] = radians(step * (360.0 / step_count))
    bpy.data.scenes["Scene"].render.filepath = '/home/b/VR/vr_shot_%d.jpg' % num
    num +=1
    bpy.ops.render.render( write_still=True )
'''
for model in models:
   bpy.ops.import_mesh.ply(filepath=model)
   obj_name = model[0:-4]
   obj = bpy.data.objects[obj_name]
   bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
   obj.location = (0,0,0)
   if obj.dimensions > Vector((35,35,35)):
       obj.scale = (0.03,0.03,0.03)
#    world_matrix = obj.matrix_world.to_translation()
   bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,size=300,location=(0,0,0),layers=((False,)+(True,)+(False,)*18))
   ico = bpy.data.objects['Icosphere']
   vertices = ico.data.vertices
   num = 0
   for v in vertices:
       cam.location = (v.co.x,v.co.y,v.co.z)
       look_at(cam, obj.matrix_world.to_translation())
       RT = get_3x4_RT_matrix_from_blender(cam)
       print(RT)
'''
