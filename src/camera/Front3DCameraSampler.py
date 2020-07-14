import random
import numpy as np

import bpy

from src.camera.CameraSampler import CameraSampler
from src.utility.BlenderUtility import get_all_mesh_objects, get_bounds

class Front3DCameraSampler(CameraSampler):

    def __init__(self, config):
        CameraSampler.__init__(self, config)
        self.used_floors = []


    def run(self):
        all_objects = get_all_mesh_objects()
        front_3D_objs = [obj for obj in all_objects if "is_3D_future" in obj and obj["is_3D_future"]]

        floor_objs = [obj for obj in front_3D_objs if obj.name.lower().startswith("floor")]

        # count objects per floor -> room
        floor_obj_counters = {obj.name: 0 for obj in floor_objs}
        counter = 0
        for obj in front_3D_objs:
            name = obj.name.lower()
            if "wall" in name or "ceiling" in name:
                continue
            counter += 1
            location = obj.location
            for floor_obj in floor_objs:
                is_above = self._position_is_above_object(location, floor_obj)
                if is_above:
                    floor_obj_counters[floor_obj.name] += 1
        amount_of_objects_needed_per_room = 2
        self.used_floors = [obj for obj in floor_objs if floor_obj_counters[obj.name] > amount_of_objects_needed_per_room]

        super().run()

    def sample_and_validate_cam_pose(self, cam, cam_ob, config):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        # Sample used floor obj
        floor_obj = random.choice(self.used_floors)

        # Sample/set intrinsics
        self._set_cam_intrinsics(cam, config)

        # Sample camera extrinsics (we do not set them yet for performance reasons)
        cam2world_matrix = self._cam2world_matrix_from_cam_extrinsics(config)

        # Make sure the sampled location is inside the room => overwrite x and y and add offset to z
        bounding_box = get_bounds(floor_obj)
        min_corner = np.min(bounding_box, axis=0)
        max_corner = np.max(bounding_box, axis=0)

        cam2world_matrix.translation[0] = random.uniform(min_corner[0], max_corner[0])
        cam2world_matrix.translation[1] = random.uniform(min_corner[1], max_corner[1])
        cam2world_matrix.translation[2] += floor_obj.location[2]

        # Check if sampled pose is valid
        if self._is_pose_valid(floor_obj, cam, cam_ob, cam2world_matrix):
            # Set camera extrinsics as the pose is valid
            cam_ob.matrix_world = cam2world_matrix
            return True
        else:
            return False


    def _is_pose_valid(self, floor_obj, cam, cam_ob, cam2world_matrix):
        """ Determines if the given pose is valid.

        - Checks if the pose is above the floor
        - Checks if the distance to objects is in the configured range
        - Checks if the scene coverage score is above the configured threshold

        :param floor_obj: The floor object of the room the camera was sampled in.
        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param cam2world_matrix: The sampled camera extrinsics in form of a camera to world frame transformation matrix.
        :return: True, if the pose is valid
        """
        if not self._position_is_above_object(cam2world_matrix.to_translation(), floor_obj):
            return False

        return super()._is_pose_valid(cam, cam_ob, cam2world_matrix)

