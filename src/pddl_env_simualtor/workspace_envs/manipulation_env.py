# write a base class call manipulation domain. Derive variants of it that build objects in specific locations
import numpy as np
import warnings

from typing import Tuple, Optional, List, Dict

import pybullet as pb

PATH_TO_LOCAL_URDF = "/home/karan-m/Documents/Research/franka_simulation/PDDLtoSim/models/"
PATH_TO_BOXES = "boxes/"


class ManipulationDomain:
    """
    A base class that builds the plane and the table in the env as these are basics of all the variants of
    manipulation domain we will be using for simulation.
    """

    def __init__(self,
                 physics_client_id,
                 workspace_lim: Optional[iter] = None):

        self._physics_client_id = physics_client_id
        self._ws_lim = tuple(workspace_lim)
        self._table_height = []
        self._objs: Dict = {}

        self.render()

    @property
    def table_height(self):
        return self._table_height

    @property
    def objs(self):
        return self._objs

    def get_object_ids(self) -> List[int]:
        """
        Every object loaded in the environment has its own uniqye id. This methods returns ONLY the list of ids.
        """

        return list(self._objs.keys())

    def get_obj_attr(self, obj_id: int) -> Tuple:
        """
        Given an object id associated with an object in the world, return its attributes such

            1) Obj name
            2) Obj current position
            3) Obj current orientation
        """

        assert type(obj_id) == int, "An object Id is always of type integer"

        if self._objs.get(obj_id) is None:
            warnings.warn(f"The object id {obj_id} is not a Valid obj id. Please enter an id from"
                          f" {self._objs.keys()}")

        _obj_attrs = self._objs.get(obj_id)
        _urdf_name = _obj_attrs[0]
        _obj_name = _obj_attrs[1]
        _obj_curr_pose = pb.getBasePositionAndOrientation(obj_id)

        return _urdf_name, _obj_name, _obj_curr_pose[0], _obj_curr_pose[1]

    def get_obj_id(self, obj_name: str):
        """
        A helper function that returns the id of the obj given an object name
        """

        for _obj_id, _obj_attrs in self.objs.items():
            if _obj_attrs[1] == obj_name:
                return _obj_id

        warnings.warn(f"Could not find an object with the name {obj_name}")
        return None

    def get_obj_ids_from(self, obj_names: list):
        """
        A helper function to get a set of object ids corresponding to a set of obj names
        """
        _obj_ids = []

        for _obj_name in obj_names:
            _obj_ids.append(self.get_obj_id(_obj_name))

        return _obj_ids

    def get_obj_pose(self, obj_name):
        """
        A help function that return the pose corresponding the object name. We first retireve the obj id and the get the
        object information in the world.
        """
        _obj_id = self.get_obj_id(obj_name)

        _obj_pos, _obj_orn = pb.getBasePositionAndOrientation(_obj_id, physicsClientId=self._physics_client_id)

        return _obj_pos, _obj_orn

    def render(self):
        # load plane
        pb.loadURDF(fileName="plane.urdf",
                    basePosition=[0, 0, 0],
                    physicsClientId=self._physics_client_id)

        # Load table and object
        self._table_id = pb.loadURDF(fileName="table/table.urdf",
                                     useFixedBase=True,
                                     physicsClientId=self._physics_client_id)

        table_info = pb.getCollisionShapeData(self._table_id, -1, physicsClientId=self._physics_client_id)[0]
        self._table_height = table_info[5][2] + table_info[3][2]/2

        # set ws limit on z according to table height
        self._ws_lim[2][:] = [self._table_height, self._table_height + 0.3]

    def load_object(self, urdf_name, obj_name, obj_init_position=None, obj_init_orientation=None):
        _filename = PATH_TO_LOCAL_URDF + PATH_TO_BOXES + urdf_name + ".urdf"

        # add the table height to the z position of the obj position
        _new_obj_pos = obj_init_position
        _new_obj_pos[2] = _new_obj_pos[2] + self.table_height

        obj_id = pb.loadURDF(fileName=_filename,
                             basePosition=_new_obj_pos,
                             baseOrientation=obj_init_orientation,
                             useFixedBase=False,
                             flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                             physicsClientId=self._physics_client_id)

        self._objs.update({obj_id: (urdf_name, obj_name, obj_init_position, obj_init_orientation)})

    def load_markers(self, marker_loc: np.ndarray, marker_len: float = 0.05, marker_width: float = 0.05):
        """
        A helper method to create visual placeholder for the blocks in the simulation world. By default we are creating
        a (very) thin (static) box with mass = 0.

        @param marker_loc: [x_pos, y_pos, z]. Keep z = 0 as we add table height by default in our simulation.
        @param marker_len: This parameter represents the half length of the shape. So the actual length is 2 times
        @param marker_width: This parameter represents the half width of the shape. So the actual length is 2 times
        """
        marker_loc[2] += self.table_height

        visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_BOX,
                                               halfExtents=[marker_len, marker_width, 0.001],
                                               rgbaColor=[0, 0, 1, 0.6],
                                               specularColor=[0.4, .4, 0],
                                               visualFramePosition=marker_loc/2,
                                               physicsClientId=self._physics_client_id)

        # collision_shape_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX,
        #                                              halfExtents=[0.05, 0.05, 0.001],
        #                                              collisionFramePosition=_loc/2,
        #                                              physicsClientId=self._physics_client_id)

        pb.createMultiBody(baseMass=0,
                           # baseInertialFramePosition=[0, 0, 0],
                           # baseCollisionShapeIndex=collision_shape_id,
                           baseVisualShapeIndex=visual_shape_id,
                           basePosition=marker_loc/2,
                           baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self._physics_client_id)

    def get_object_shape_info(self, obj_id):
        info = list(pb.getCollisionShapeData(obj_id, -1, physicsClientId=self._physics_client_id)[0])
        info[4] = pb.getVisualShapeData(obj_id, -1, physicsClientId=self._physics_client_id)[0][4]
        return info
