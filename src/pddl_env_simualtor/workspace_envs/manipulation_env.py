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
        self._obj_ids: Dict = {}

        self.render()

    @property
    def table_height(self):
        return self.table_height

    @property
    def obj_ids(self):
        return self._obj_ids

    def get_object_ids(self) -> List[int]:
        """
        Every object loaded in the environment has its own uniqye id. This methods returns ONLY the list of ids.
        """

        return list(self._obj_ids.keys())

    def get_obj_attr(self, obj_id: int) -> Tuple:
        """
        Given an object id associated with an object in the world, return its attributes such

            1) Obj name
            2) Obj current position
            3) Obj current orientation
        """

        assert type(obj_id) == int, "An object Id is always of type integer"

        if self._obj_ids.get(obj_id) is None:
            warnings.warn(f"The object id {obj_id} is not a Valid obj id. Please enter an id from"
                          f" {self._obj_ids.keys()}")

        _obj_attrs = self._obj_ids.get(obj_id)
        _obj_name = _obj_attrs[0]
        _obj_curr_pose = pb.getBasePositionAndOrientation(obj_id)

        return _obj_name, _obj_curr_pose[0], _obj_curr_pose[1]

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

    def load_object(self, obj_name, obj_init_position=None, obj_init_orientation=None):
        _filename = PATH_TO_LOCAL_URDF + PATH_TO_BOXES + obj_name + ".urdf"

        obj_id = pb.loadURDF(fileName=_filename,
                             basePosition=obj_init_position,
                             baseOrientation=obj_init_orientation,
                             useFixedBase=False,
                             flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                             physicsClientId=self._physics_client_id)

        self._obj_ids.update({obj_id: (obj_name, obj_init_position, obj_init_orientation)})

    def get_object_shape_info(self, obj_id):
        info = list(pb.getCollisionShapeData(obj_id, -1, physicsClientId=self._physics_client_id)[0])
        info[4] = pb.getVisualShapeData(obj_id, -1, physicsClientId=self._physics_client_id)[0][4]
        return info

