import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('scene_description_npz', nargs='?', help="Path to scene description")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

d = np.load(args.scene_description_npz)
K = d["K"]
poses = d["object_poses"]
cam_pose = d["camera_pose"]
box_dims = d["box_dims"]
im_width, im_height = d["image_dims"]
floor_material = d["floor_material"]
wall_material = d["wall_material"]
object_materials = d["object_materials"]
(room_x_half_width, room_z_half_width) = d["room_half_widths"]
floor_y = d["floor_y"]

num_objects = poses.shape[0]

bproc.init()
objects = [
    bproc.object.create_primitive("CUBE", scale=[box_dims[i,0]/2,box_dims[i,1]/2,box_dims[i,2]/2])
    for i in range(num_objects)
]

# set shading and physics properties and randomize PBR materials
for j, obj in enumerate(objects):
    obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    obj.set_shading_mode('auto')
    obj.set_local2world_mat(
        poses[j,:,:]
    )
        
floor_plane = [
    bproc.object.create_primitive('PLANE',
    scale=[room_x_half_width*2, room_z_half_width*2, 1], location=[0, floor_y, 0], rotation=[np.pi/2, 0, 0]),
]
wall_planes = [
    bproc.object.create_primitive('PLANE', scale=[room_x_half_width*2, 2, 1], location=[0, 0, -room_z_half_width], rotation=[0, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, room_z_half_width*2, 1], location=[room_x_half_width, 0, 0], rotation=[0, np.pi/2, 0]),
    bproc.object.create_primitive('PLANE', scale=[room_x_half_width*2, 2, 1], location=[0, 0, room_z_half_width], rotation=[0, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, room_z_half_width*2, 1], location=[-room_x_half_width, 0, 0], rotation=[0, np.pi/2, 0]),
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
texture_name = floor_material
tex = bproc.filter.one_by_attr(cc_textures, "name", texture_name)
for plane in floor_plane:
    plane.replace_materials(tex)

texture_name = wall_material
tex = bproc.filter.one_by_attr(cc_textures, "name", texture_name)
for plane in wall_planes:
    plane.replace_materials(tex)

texture_names = object_materials
for (i, obj) in enumerate(objects):
    tex = bproc.filter.one_by_attr(cc_textures, "name", texture_names[i])
    obj.replace_materials(tex)

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
