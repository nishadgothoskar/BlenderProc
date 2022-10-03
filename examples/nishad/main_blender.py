import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('scene_description_npz', nargs='?', help="Path to scene description")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
args = parser.parse_args()

d = np.load(args.scene_description_npz)
K = d["K"]
poses = d["object_poses"]
cam_pose = d["camera_pose"]
object_model_paths = d["object_model_paths"]
im_width, im_height = d["image_dims"]

num_objects = poses.shape[0]

bproc.init()

# set shading and physics properties and randomize PBR materials
print(num_objects)
for i in range(num_objects):
    p = object_model_paths[i]
    if p.endswith(".blend"):
        objs = bproc.loader.load_blend(p)
    else:
        objs = bproc.loader.load_obj(p)
    print(objs)
    for obj in objs:
        # obj.enable_rigidbody(True, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        # obj.set_shading_mode('auto')
        obj.set_local2world_mat(
            poses[i,:,:].dot(obj.get_local2world_mat())
        )
        print(obj.get_local2world_mat())

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -10, 0])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=1000.0, 
                                   emission_color=np.random.uniform([0.9, 0.9, 0.9, 1.0], [1.0, 1.0, 1.0, 1.0]))    
light_plane.replace_materials(light_plane_material)

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(1000)
light_point.set_color(np.random.uniform([0.9,0.9,0.9],[1,1,1]))
light_point.set_location([0.0, -10.0, 40.0])


# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(K,im_width, im_height)

# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_pose, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.set_output_format(enable_transparency=True)

# render the whole pipeline
data = bproc.renderer.render()

bproc.writer.write_hdf5(args.output_dir, data)
