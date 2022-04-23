import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

d = np.load("/home/nishadg/mit/CoraAgent/scene.npy.npz")
K = d["K"]
poses = d["poses"]
cam_pose = d["camera"]
box_dims = d["box_dims"]
im_width, im_height = d["image_dims"]
print(K)
print(box_dims)
print(box_dims.shape)
print(poses.shape)
num_objects = poses.shape[0]

print(poses[0,:,:])
print(poses[1,:,:])
print(poses[2,:,:])
print(poses[3,:,:])
print(poses[4,:,:])

bproc.init()
objects = [
    bproc.object.create_primitive("CUBE", scale=[box_dims[i,0]/2,box_dims[i,1]/2,box_dims[i,2]/2])
    for i in range(num_objects)
]
print(len(objects))

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(objects):
    obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    obj.set_shading_mode('auto')
    obj.set_local2world_mat(
        poses[j,:,:]
    )
        
floor_plane = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0.5 - 0.025, 0], rotation=[np.pi/2, 0, 0]),
]
wall_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0, -2], rotation=[0, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 0], rotation=[0, np.pi/2, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0, 2], rotation=[0, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 0], rotation=[0, np.pi/2, 0]),
]
for plane in (floor_plane + wall_planes):
    plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -4, 0])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=10.0, 
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)


# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(500)
light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
light_point.set_location([0.0, -1.0, 0.0])


# sample CC Texture and assign to room planes
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
print([c.get_attr("name") for c in cc_textures])
texture_name = "Wood081"
tex = bproc.filter.one_by_attr(cc_textures, "name", texture_name)
for plane in floor_plane:
    plane.replace_materials(tex)

texture_name = "Plaster001"
tex = bproc.filter.one_by_attr(cc_textures, "name", texture_name)
for plane in wall_planes:
    plane.replace_materials(tex)

texture_names = ["Metal030" for _ in range(len(objects))] 
for (i, obj) in enumerate(objects):
    tex = bproc.filter.one_by_attr(cc_textures, "name", texture_names[i])
    obj.replace_materials(tex)


# # Define a function that samples 6-DoF poses
# def sample_pose_func(obj: bproc.types.MeshObject):
#     min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
#     max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
#     obj.set_location(np.random.uniform(min, max))
#     obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(K,im_width, im_height)


# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_pose, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world)


# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# render the whole pipeline
data = bproc.renderer.render()

bproc.writer.write_hdf5(args.output_dir, data)

import h5py
import os

f = h5py.File(
    os.path.join('/home/nishadg/mit/BlenderProc/examples/nishad/output/0.hdf5'),
    'r'
)