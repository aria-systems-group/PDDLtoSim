'''
This script contains the code to execute a given strategy using the builtin PyBullet simulator. We either run a strategy or load a saved simulation. 
'''

import re
import sys
import warnings
import math
import copy

import numpy as np
import pybullet as pb

from typing import Dict, List, Tuple
from utls import deprecated
from src.pddl_env_simualtor.envs.panda_sim import PandaSim

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem


def get_action_from_causal_graph_edge(causal_graph_edge_str: str) -> str:
    """
    A function to extract the appropriate action type given an edge string (a valid action) on the causal graph.
    Currently the valid action types are:

        1. transit
        2. transfer
        3. grasp
        4. release
        5. human-move
    """
    _transit_pattern = "\\btransit\\b"
    _transfer_pattern = "\\btransfer\\b"
    _grasp_pattern = "\\bgrasp\\b"
    _release_pattern = "\\brelease\\b"
    _human_move_pattern = "\\bhuman-move\\b"

    if re.search(_transit_pattern, causal_graph_edge_str):
        return "transit"

    if re.search(_transfer_pattern, causal_graph_edge_str):
        return "transfer"

    if re.search(_grasp_pattern, causal_graph_edge_str):
        return "grasp"

    if re.search(_release_pattern, causal_graph_edge_str):
        return "release"

    if re.search(_human_move_pattern, causal_graph_edge_str):
        return "human-move"

    warnings.warn("The current string does not have valid action type")
    sys.exit(-1)


def get_multiple_box_location(multiple_box_location_str: str) -> Tuple[int, List[str]]:
    """
    A function that return multiple locations (if present) in a str.

    In our construction of transition system, as per our pddl file naming convention, a human action is as follows
    "human-action b# l# l#", the box # is placed on l# (1st one) and the human moves it to l# (2nd one).
    """

    _loc_pattern = "[l|L][\d]+"
    try:
        _loc_states: List[str] = re.findall(_loc_pattern, multiple_box_location_str)
    except AttributeError:
        print(f"The causal_state_string {multiple_box_location_str} dose not contain location of the box")

    _box_pattern = "[b|B][\d]+"
    try:
        _box_state: str = re.search(_box_pattern, multiple_box_location_str).group()
    except AttributeError:
        print(f"The causal_state_string {multiple_box_location_str} dose not contain box id")

    _box_id_pattern = "\d+"
    _box_id: int = int(re.search(_box_id_pattern, _box_state).group())

    return _box_id, _loc_states


def re_arrange_blocks(box_id: int, curr_loc, sim_handle):
    """
    A function to place all the blocks at their respective locations.
    """

    # for _box_id, _box_loc in current_world_confg:
    #     if _box_loc != "gripper" and _box_id <= 2:
    _obj_name = f"b{box_id}"
    _obj_id = sim_handle.world.get_obj_id(_obj_name)
    _urdf_name, _, _, _ = sim_handle.world.get_obj_attr(_obj_id)
    pb.removeBody(_obj_id)

    # you have to subtract the table height
    curr_loc[2] = curr_loc[2] - sim_handle.world.table_height

    # add a new to the location that human moved-the obj too
    sim_handle.world.load_object(urdf_name=_urdf_name,
                                 obj_name=_obj_name,
                                 obj_init_position=curr_loc,
                                 obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))


def initialize_saved_simulation(record_sim: bool,
                                boxes: list,
                                box_locs: list,
                                init_conf: list,
                                loc_dict: dict,
                                debug: bool = False):
    # obj to URDF mapping for diag case
    # _box_id_to_urdf = {
    #     # "b0": "black_box",
    #     # "b1": "grey_box",
    #     # "b2": "white_box"
    #     "b0": "black_box",
    #     "b1": "grey_box",
    #     "b2": "white_box",
    #     "b3": "black_box",
    #     "b4": "grey_box",
    # }

    _box_id_to_urdf = {
        "b0": "white_box",
        "b1": "black_box",
        "b2": "black_box",
        "b3": "black_box",
        "b4": "black_box",
    }

    # build the simulator
    if record_sim:
        physics_client = pb.connect(pb.GUI,
                                    options="--minGraphicsUpdateTimeMs=0 --mp4=\"experiment.mp4\" --mp4fps=240")
    else:
        physics_client = pb.connect(pb.GUI)

    panda = PandaSim(physics_client, use_IK=1)

    if debug:
        print(f"# of boxes = {len(boxes)}; # of locs = {len(box_locs)}")

    # initialize objects at the corresponding locs
    for _idx, _loc in enumerate(init_conf):
        if _idx == len(init_conf) - 1:
            continue
        _urdf_name = _box_id_to_urdf.get(f"b{_idx}")
        panda.world.load_object(urdf_name=_urdf_name,
                                obj_name=f"b{_idx}",
                                obj_init_position=copy.copy(loc_dict.get(_loc)),
                                obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

    for _loc in box_locs:
        _visual_marker_loc = copy.copy(loc_dict.get(_loc))
        _visual_marker_loc[2] = 0
        panda.world.load_markers(marker_loc=_visual_marker_loc)

    return panda




def initialize_simulation(causal_graph: CausalGraph,
                          transition_system: FiniteTransitionSystem,
                          loc_dict: dict,
                          record_sim: bool = False,
                          debug: bool = False):
    # obj to URDF mapping for diag case
    _box_id_to_urdf = {
        "b0": "black_box",
        "b1": "grey_box",
        "b2": "white_box",
        "b3": "black_box",
        # "b4": "white_box"
    }

    # _box_id_to_urdf = {
    #     "b0": "white_box",
    #     "b1": "black_box",
    #     "b2": "black_box",
    #     "b3": "black_box",
    #     "b4": "black_box",
    # }

    # build the simulator
    if record_sim:
        physics_client = pb.connect(pb.GUI,
                                    options="--minGraphicsUpdateTimeMs=0 --mp4=\"experiment.mp4\" --mp4fps=240")
    else:
        physics_client = pb.connect(pb.GUI)

    panda = PandaSim(physics_client, use_IK=1)

    # get the number of boxes and their locations from the pddl file.
    boxes = causal_graph.task_objects
    box_locs = causal_graph.task_locations

    if debug:
        print(f"# of boxes = {len(boxes)}; # of locs = {len(box_locs)}")

    # initialize objects at the corresponding locs
    _init_state = transition_system.transition_system.get_initial_states()[0][0]
    _init_conf = transition_system.transition_system.get_state_w_attribute(_init_state, "list_ap")

    for _idx, _loc in enumerate(_init_conf):
        if _idx == len(_init_conf) - 1:
            continue
        _urdf_name = _box_id_to_urdf.get(f"b{_idx}")
        panda.world.load_object(urdf_name=_urdf_name,
                                obj_name=f"b{_idx}",
                                obj_init_position=copy.copy(loc_dict.get(_loc)),
                                obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

    for _loc in box_locs:
        _visual_marker_loc = copy.copy(loc_dict.get(_loc))
        _visual_marker_loc[2] = 0
        panda.world.load_markers(marker_loc=_visual_marker_loc)

    return panda


def load_pre_built_loc_info(exp_name: str) -> Dict[str, np.ndarray]:
    if exp_name == "diag":
        _loc_dict = {
            # 'l0': np.array([-0.7, -0.2, 0.17/2]),
            # 'l2': np.array([-0.4, 0.2, 0.17/2]),
            # 'l1': np.array([-0.4, -0.2, 0.17/2]),
            # 'l3': np.array([0.6, -0.2, 0.17/2]),
            # 'l4': np.array([0.45, 0.0, 0.17/2])
            'l0': np.array([-0.5, -0.2, 0.17 / 2]),
            'l2': np.array([-0.5, 0.2, 0.17 / 2]),
            'l3': np.array([-0.3, -0.2, 0.17 / 2]),
            'l1': np.array([-0.3, 0.2, 0.17 / 2]),
            'l4': np.array([-0.4, 0.0, 0.17 / 2]),
            'l5': np.array([0.6, -0.2, 0.17 / 2]),
            'l6': np.array([0.4, 0.2, 0.17 / 2]),
            'l9': np.array([0.5, 0.0, 0.17 / 2]),
            # 'l8': np.array([0.3, -0.2, 0.17 / 2]),
            'l8': np.array([0.0, 0.0, 0.17 / 2]),
            # 'l6': np.array([0.3, 0.2, 0.17 / 2]),
            'l7': np.array([-0.1, 0.0, 0.17 / 2]),
            # 'l9': np.array([0.4, 0.0, 0.17 / 2]),
            # 'l7': np.array([0.3, 0.0, 0.17 / 2]),
        }
    elif exp_name == "arch":
        # both locations are on the top
        _loc_dict = {
            'l0': np.array([0.5, 0.0, 0.0]),
            'l1': np.array([-0.5, 0.0, 0.0]),
            'l2': np.array([-0.4, -0.14/2, 0.17/2]),
            # 'l2': np.array([-0.6, -0.2, 0.17 / 2]),
            'l3': np.array([-0.4, 0.14/2, 0.17/2]),
            'l4': np.array([-0.2, 0.0, 0.17/2]),
            'l5': np.array([0.0, -0.3, 0.17/2]),
            # 'l6': np.array([0.0, +0.3, 0.17/2]),
            # 'l7': np.array([0.3, 0.0, 0.17/2]),
            'l6': np.array([-0.1, 0.0, 0.17 / 2]),
            'l7': np.array([0.1, 0.0, 0.17 / 2]),
            # 'l7': np.array([0.6, 0.0, 0.17 / 2]),
            'l8': np.array([0.5, 0.14/2, 0.17/2]),
            'l9': np.array([0.5, -0.14/2, 0.17/2]),
        }
    else:
        _loc_dict: dict = {}
        warnings.warn("PLease enter a valid experiment name")

    return _loc_dict



def _pre_loaded_pick_and_place_action(pos, panda):
    _wait_pos_left = [-0.2, 0.0, 0.9, math.pi, 0, math.pi]
    _wait_pos_right = [0.2, 0.0, 0.9, math.pi, 0, math.pi]

    # lets try grabbing from side and place on top of two objects
    # give pose and orientation
    panda.apply_high_level_action("openEE", [])

    panda.apply_high_level_action("transit", _wait_pos_left, vel=0.5)

    _pos = [pos[0], pos[1], pos[2] + 0.3, 0, math.pi/2, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)

    # # 2. go down towards the object and grab it
    _pos = [pos[0], pos[1], pos[2], 0, math.pi/2, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)

    panda.apply_high_level_action("closeEE", [], vel=0.5)

    # # 3. grab the object and take the appr stance
    _pos = [pos[0], pos[1], pos[2] + 0.3,  math.pi, 0, 0]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)

    _pos = [pos[0], pos[1], pos[2] + 0.3, math.pi, 0, math.pi/2]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)

    # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
    _pos = [-0.5, 0.0, 0.625 + 0.20, math.pi, 0, math.pi / 2]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)

    _pos = [-0.5, 0.0, 0.625 + 0.17, math.pi, 0, math.pi/2]
    panda.apply_high_level_action("transfer", _pos, vel=0.25)

    panda.apply_high_level_action("openEE", [], vel=0.25)

    _pos = [-0.5, 0.0, 0.625 + 0.30, math.pi, 0, math.pi / 2]
    panda.apply_high_level_action("transit", _pos, vel=0.25)


@deprecated
def execute_str(actions: list,
                causal_graph: CausalGraph,
                transition_system: FiniteTransitionSystem,
                exp_name: str,
                record_sim: bool = False,
                debug: bool = False):
    # determine the action type first
    _action_type = ""
    _loc_dict: dict = load_pre_built_loc_info(exp_name=exp_name)

    # some constants useful during simulation
    _wait_pos_left = [-0.2, 0.0, 1.2, math.pi, 0, math.pi]
    _wait_pos_right = [0.2, 0.0, 1.2, math.pi, 0, math.pi]

    # load the simulator env
    panda_handle = initialize_simulation(causal_graph=causal_graph,
                                         transition_system=transition_system,
                                         loc_dict=_loc_dict,
                                         record_sim=record_sim,
                                         debug=debug)

    # loop and add the const table height to all valid loc
    for _loc in _loc_dict.values():
        _loc[2] = _loc[2] + panda_handle.world.table_height

    _release_from_top_loc_to_right = False
    _release_from_top_loc_to_left = False

    for _action in actions:
        _action_type = transition_system._get_action_from_causal_graph_edge(_action)
        _box_id, _loc = transition_system._get_multiple_box_location(_action)
        # _current
        if len(_loc) == 2:
            _from_loc = _loc[0]
            _to_loc = _loc[1]
        else:
            _from_loc = ""
            _to_loc = _loc[0]
        _loc = _loc_dict.get(_to_loc)
        _org_loc_copy = copy.copy(_loc)

        # if you building an arch then lo and l1 are location on top of boxes. You need a different type of grab action
        # to execute this successfully.
        _transfer_to_top_loc: bool = False
        if _to_loc in ["l0", "l1"]:
            _transfer_to_top_loc = True

        _transfer_from_top_loc = False
        if _from_loc in ["l0", "l1"]:
            _transfer_from_top_loc = True

        if _action_type == "transit":
            # pre-image based on the object loc
            if _loc[0] < 0:
                panda_handle.apply_high_level_action("transit", _wait_pos_left, vel=0.5)
            else:
                panda_handle.apply_high_level_action("transit", _wait_pos_right, vel=0.5)
            # every transfer and transit action will have a from and to location. Lets extract it.
            _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]

            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

        elif _action_type == "transfer":
            if _transfer_to_top_loc:
                # place it at an intermediate loc and grab it from side and then continue
                panda_handle.apply_high_level_action("transfer", [0.0, 0.0, 0.65 + 0.17, math.pi, 0, math.pi], vel=0.5)
                panda_handle.apply_high_level_action("openEE", [], vel=0.5)
                # grab it from side based on where you are going
                if _loc[0] < 0:
                    # going left
                    _pos = [+0.1, 0.0, 0.625 + 0.2 + 0.4, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    # _pos = [+0.1, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    # 2. go down towards the object and grab it
                    _pos = [0.03, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    _pos = [-0.01, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

                    # 3. grab the object and take the appr stance
                    _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _org_loc = copy.copy(_loc)
                    # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
                    _pos = [_loc[0], _loc[1], _loc[2] + 0.20, 0, math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, 0, math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)

                else:
                    # going right
                    _pos = [-0.1, 0.0, 0.625 + 0.2 + 0.4, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    # 2. go down towards the object and grab it
                    _pos = [-0.03, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    _pos = [0.0, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

                    # 3. grab the object and take the appr stance
                    _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, math.pi/2, -math.pi]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _org_loc = copy.copy(_loc)
                    # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
                    _pos = [_loc[0], _loc[1], _loc[2] + 0.20, 0,  math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, 0, math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)

            elif _transfer_from_top_loc:
                # pre-image
                if _loc[0] < 0:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)

                    _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, math.pi/2, -math.pi]

                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
                    _release_from_top_loc_to_left = True

                else:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

                    _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, math.pi/2, math.pi]

                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
                    _release_from_top_loc_to_right = True

            else:
                # pre-image
                if _loc[0] < 0:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)
                else:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

                _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

        elif _action_type == "grasp":
            # pre-image
            _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

            panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

            # move up
            _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

        elif _action_type == "release":
            # release action is different when dropping at top location
            if _transfer_to_top_loc:
                panda_handle.apply_high_level_action("openEE", [], vel=0.25)

                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, math.pi / 2]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)
                break

            elif _release_from_top_loc_to_right:
                _release_from_top_loc_to_right = False

                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, math.pi/2, math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)
                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, math.pi / 2]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)

                # _pos = [_loc[0], _loc[1], _loc[2] - 0.3 - 0.05]
                re_arrange_blocks(box_id=_box_id, curr_loc=np.array(_org_loc_copy), sim_handle=panda_handle)

            elif _release_from_top_loc_to_left:
                _release_from_top_loc_to_left = False
                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, math.pi/2, -math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)
                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, -math.pi]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)
            else:

                # pre-image
                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)

                #post_image
                _pos = [_loc[0], _loc[1], _loc[2] + 3, math.pi, 0, math.pi]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                # _pos = [_loc[0], _loc[1], _loc[2] - 3 - 0.05]
                re_arrange_blocks(box_id=_box_id, curr_loc=np.array(_org_loc_copy), sim_handle=panda_handle)

        elif _action_type == "human-move":
            # get the urdf name and remove the existing body
            _obj_name = f"b{_box_id}"
            _obj_id = panda_handle.world.get_obj_id(_obj_name)
            _urdf_name, _, _, _ = panda_handle.world.get_obj_attr(_obj_id)
            pb.removeBody(_obj_id)

            # you have to subtract the table height
            _loc[2] = _loc[2] - panda_handle.world.table_height

            # add a new to the location that human moved-the obj too
            panda_handle.world.load_object(urdf_name=_urdf_name,
                                           obj_name=_obj_name,
                                           obj_init_position=_loc,
                                           obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

        else:
            warnings.warn(f"The current action {_action} does not have a valid action type")
            sys.exit(-1)




def execute_saved_str(yaml_data: dict,
                      exp_name: str,
                      record_sim: bool = False,
                      debug: bool = False):
    """
    A helper function to execute a saved simulation.
    """
    # determine the action type first
    _action_type = ""
    _loc_dict = load_pre_built_loc_info(exp_name=exp_name)
    actions = yaml_data.get("reg_str")

    # some constants useful during simulation
    _wait_pos_left = [-0.2, 0.0, 1.2, math.pi, 0, -math.pi]
    _wait_pos_right = [0.2, 0.0, 1.2, math.pi, 0, math.pi]

    _boxes = yaml_data["no_of_boxes"].get("objects")
    _box_locs = yaml_data["no_of_loc"].get("objects")
    _init_conf = yaml_data.get("init_worl_conf")

    # load the simulator env
    panda_handle = initialize_saved_simulation(record_sim=record_sim,
                                               boxes=_boxes,
                                               box_locs=_box_locs,
                                               init_conf=_init_conf,
                                               loc_dict=_loc_dict,
                                               debug=debug)

    # # loop and add the const table height to all valid loc
    # for _loc in _loc_dict.values():
    #     _loc[2] = _loc[2] + panda_handle.world.table_height
    #
    # for _action in actions:
    #     _action_type = get_action_from_causal_graph_edge(_action)
    #     _box_id, _loc = get_multiple_box_location(_action)
    #     if len(_loc) == 2:
    #         _from_loc: str = _loc[0]
    #         _to_loc: str = _loc[1]
    #     else:
    #         _from_loc: str = ""
    #         _to_loc: str = _loc[0]
    #     _loc = _loc_dict.get(_to_loc)
    #
    #     # if you building an arch then lo and l1 are location on top of boxes. You need a different type of grab action
    #     # to execute this successfully.
    #     _transfer_to_top_loc: bool = False
    #     if _to_loc in ["l0", "l1"]:
    #         _transfer_to_top_loc = True
    #
    #     _transfer_from_top_loc = False
    #     if _from_loc in ["l0", "l1"]:
    #         _transfer_from_top_loc = True
    #
    #     if _action_type == "transit":
    #         # pre-image based on the object loc
    #         if _loc[0] < 0:
    #             panda_handle.apply_high_level_action("transit", _wait_pos_left, vel=0.5)
    #         else:
    #             panda_handle.apply_high_level_action("transit", _wait_pos_right, vel=0.5)
    #
    #         _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
    #
    #         panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #     elif _action_type == "transfer":
    #         if not _transfer_to_top_loc:
    #             # pre-image
    #             if _loc[0] < 0:
    #                 panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)
    #             else:
    #                 panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
    #
    #             _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
    #
    #             panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #         elif _transfer_to_top_loc:
    #             # place it at an intermediate loc and grab it from side and then continue
    #             panda_handle.apply_high_level_action("transfer", [0.0, 0.0, 0.625 + 0.2, math.pi, 0, math.pi], vel=0.5)
    #             panda_handle.apply_high_level_action("openEE", [], vel=0.5)
    #             # grab it from side based on where you are going
    #             if _loc[0] < 0:
    #                 # going left
    #                 _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, math.pi / 2, -math.pi]
    #                 panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #                 # # 2. go down towards the object and grab it
    #                 _pos = [0.0, 0.0, 0.625 + 0.17/2, 0, math.pi / 2, -math.pi]
    #                 panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #             else:
    #                 # going right
    #                 _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, math.pi / 2, math.pi]
    #                 panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #                 # # 2. go down towards the object and grab it
    #                 _pos = [0.0, 0.0, 0.625 + 0.17/2, 0, math.pi / 2, math.pi]
    #                 panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #             panda_handle.apply_high_level_action("closeEE", [], vel=0.5)
    #
    #             # # 3. grab the object and take the appr stance
    #             _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, math.pi, 0, 0]
    #             panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #             # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, math.pi, 0, math.pi / 2]
    #             # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #             # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
    #             _org_loc = _loc
    #             _pos = [_loc[0], _loc[1], _loc[2] + 0.20, math.pi, 0, math.pi / 2]
    #             panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #             _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, math.pi, 0, math.pi / 2]
    #             panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)
    #
    #         elif _transfer_from_top_loc:
    #             # pre-image
    #             if _loc[0] < 0:
    #                 panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)
    #
    #                 _pos = [_loc[0], _loc[1], _loc[2] + 0.3, -math.pi, math.pi / 2, math.pi]
    #
    #                 panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #             else:
    #                 panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
    #
    #                 _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, math.pi/2, math.pi]
    #
    #                 panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #         else:
    #             warnings.warn("Encountered an error in type of transfer action being executed")
    #
    #     elif _action_type == "grasp":
    #         # pre-image
    #         _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
    #         panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #         panda_handle.apply_high_level_action("closeEE", [], vel=0.5)
    #
    #         # move up
    #         _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
    #         panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #     elif _action_type == "release":
    #         # release action is different when dropping at top location
    #         if _transfer_to_top_loc:
    #             panda_handle.apply_high_level_action("openEE", [], vel=0.25)
    #
    #             _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, math.pi / 2]
    #             panda_handle.apply_high_level_action("transit", _pos, vel=0.25)
    #
    #         # pre-image
    #         _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
    #         panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
    #
    #         panda_handle.apply_high_level_action("openEE", [], vel=0.5)
    #
    #         # post_image
    #         _pos = [_loc[0], _loc[1], _loc[2] + 3, math.pi, 0, math.pi]
    #         panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
    #
    #     elif _action_type == "human-move":
    #         # get the urdf name and remove the existing body
    #         _obj_name = f"b{_box_id}"
    #         _obj_id = panda_handle.world.get_obj_id(_obj_name)
    #         _urdf_name, _, _, _ = panda_handle.world.get_obj_attr(_obj_id)
    #         pb.removeBody(_obj_id)
    #
    #         # you have to subtract the table height
    #         _loc[2] = _loc[2] - panda_handle.world.table_height
    #
    #         # add a new to the location that human moved-the obj too
    #         panda_handle.world.load_object(urdf_name=_urdf_name,
    #                                        obj_name=_obj_name,
    #                                        obj_init_position=_loc,
    #                                        obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))
    #
    #     else:
    #         warnings.warn(f"The current action {_action} does not have a valid action type")
    #         sys.exit(-1)
    # loop and add the const table height to all valid loc
    for _loc in _loc_dict.values():
        _loc[2] = _loc[2] + panda_handle.world.table_height

    _release_from_top_loc_to_right = False
    _release_from_top_loc_to_left = False

    for _action in actions:
        _action_type = get_action_from_causal_graph_edge(_action)
        _box_id, _loc = get_multiple_box_location(_action)
        # _current
        if len(_loc) == 2:
            _from_loc = _loc[0]
            _to_loc = _loc[1]
        else:
            _from_loc = ""
            _to_loc = _loc[0]
        _loc = _loc_dict.get(_to_loc)
        _org_loc_copy = copy.copy(_loc)

        # if you building an arch then lo and l1 are location on top of boxes. You need a different type of grab action
        # to execute this successfully.
        if _to_loc == "l7":
            # [-0.1, 0.0, 0.17 / 2]
            # panda_handle.apply_high_level_action("transfer", [0.0, 0.0, 0.65 + 0.17, math.pi, 0, math.pi], vel=0.5)
            # going right
            _pos = [0.1-0.1, 0.0, 0.625 + 0.2 + 0.4, 0, -math.pi / 2, -math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=1)

            # 2. go down towards the object and grab it
            _pos = [0.1-0.03, 0.0, 0.625 + 0.17 / 2, 0, -math.pi / 2, -math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

            _pos = [0.1+0.0, 0.0, 0.625 + 0.17 / 2, 0, -math.pi / 2, -math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

            panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

            # 3. grab the object and take the appr stance
            _pos = [0.1, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi, -math.pi]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

            # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, -math.pi]
            # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

            _loc = np.array([-0.4, 0.0, 0.625])
            _org_loc = copy.copy(_loc)
            # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
            _pos = [_loc[0], _loc[1], _loc[2] + 0.20, 0, -math.pi, -math.pi / 2]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

            _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, 0, -math.pi, -math.pi / 2]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)

            panda_handle.apply_high_level_action("openEE", [], vel=0.5)
            _pos = [_loc[0], _loc[1], _loc[2] + 0.30, -math.pi, 0, -math.pi / 2]
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)
            break


        _transfer_to_top_loc: bool = False
        if _to_loc in ["l0", "l1"]:
            _transfer_to_top_loc = True

        _transfer_from_top_loc = False
        if _from_loc in ["l0", "l1"]:
            _transfer_from_top_loc = True

        if _action_type == "transit":
            # pre-image based on the object loc
            if _loc[0] < 0:
                panda_handle.apply_high_level_action("transit", _wait_pos_left, vel=0.5)
            else:
                panda_handle.apply_high_level_action("transit", _wait_pos_right, vel=0.5)
            # every transfer and transit action will have a from and to location. Lets extract it.
            _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]

            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

        elif _action_type == "transfer":
            if _transfer_to_top_loc:
                # place it at an intermediate loc and grab it from side and then continue
                panda_handle.apply_high_level_action("transfer", [0.0, 0.0, 0.65 + 0.17, math.pi, 0, math.pi], vel=0.5)
                # time.sleep(3)
                panda_handle.apply_high_level_action("openEE", [], vel=0.5)

                # grab it from side based on where you are going
                if _loc[0] < 0:
                    # going left
                    _pos = [+0.1, 0.00, 0.625 + 0.2 + 0.4, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    re_arrange_blocks(box_id=_box_id, curr_loc=np.array([0.0, 0.0, 0.65 + 0.17 / 2]),
                                      sim_handle=panda_handle)

                    # _pos = [+0.1, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    # 2. go down towards the object and grab it
                    _pos = [0.03, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.25)

                    _pos = [-0.01, 0.0, 0.625 + 0.17 / 2, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

                    # 3. grab the object and take the appr stance
                    _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, math.pi / 2, -math.pi]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _org_loc = copy.copy(_loc)
                    # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
                    _pos = [_loc[0], _loc[1], _loc[2] + 0.20, 0, math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, 0, math.pi, -math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)

                else:
                    # going right
                    _pos = [-0.1, 0.0, 0.625 + 0.2 + 0.4, 0, -math.pi / 2, math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=1)

                    re_arrange_blocks(box_id=_box_id, curr_loc=np.array([0.0, 0.0, 0.65 + 0.17 / 2]),
                                      sim_handle=panda_handle)

                    # 2. go down towards the object and grab it
                    _pos = [-0.03, 0.0, 0.625 + 0.17 / 2, 0, -math.pi / 2, math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    _pos = [0.0, 0.0, 0.625 + 0.17 / 2, 0, -math.pi / 2, math.pi]
                    panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                    panda_handle.apply_high_level_action("closeEE", [], vel=0.5)

                    # 3. grab the object and take the appr stance
                    _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, math.pi]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    # _pos = [0.0, 0.0, 0.625 + 0.2 + 0.3, 0, -math.pi / 2, -math.pi]
                    # panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _org_loc = copy.copy(_loc)
                    # panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
                    _pos = [_loc[0], _loc[1], _loc[2] + 0.20, 0, -math.pi, math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                    _pos = [_org_loc[0], _org_loc[1], _org_loc[2] + 0.17, 0, -math.pi, math.pi / 2]
                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.25)

            elif _transfer_from_top_loc:
                # pre-image
                if _loc[0] < 0:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)

                    _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, math.pi / 2, -math.pi]

                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
                    _release_from_top_loc_to_left = True

                else:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

                    _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, math.pi / 2, math.pi]

                    panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)
                    _release_from_top_loc_to_right = True

            else:
                # pre-image
                if _loc[0] < 0:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)
                else:
                    panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

                _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

        elif _action_type == "grasp":
            # pre-image
            _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=1.0)

            panda_handle.apply_high_level_action("closeEE", [], vel=1.0)

            # move up
            _pos = [_loc[0], _loc[1], _loc[2] + 0.4, math.pi, 0, math.pi]
            # _pos = [_loc[0], _loc[1], _loc[2] + 0.7, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

        elif _action_type == "release":
            # release action is different when dropping at top location
            if _transfer_to_top_loc:
                panda_handle.apply_high_level_action("openEE", [], vel=0.25)

                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, math.pi / 2]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)
                break

            elif _release_from_top_loc_to_right:
                _release_from_top_loc_to_right = False

                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, math.pi / 2, math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)
                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, math.pi / 2]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)

                # _pos = [_loc[0], _loc[1], _loc[2] - 0.3 - 0.05]
                re_arrange_blocks(box_id=_box_id, curr_loc=np.array(_org_loc_copy), sim_handle=panda_handle)

            elif _release_from_top_loc_to_left:
                _release_from_top_loc_to_left = False
                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, math.pi / 2, -math.pi]

                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)
                _pos = [_loc[0], _loc[1], _loc[2] + 0.30, math.pi, 0, -math.pi]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.25)

                _org_loc_copy[0] = _org_loc_copy[0] + 0.02
                re_arrange_blocks(box_id=_box_id, curr_loc=np.array(_org_loc_copy), sim_handle=panda_handle)
            else:

                # pre-image
                _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
                panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

                panda_handle.apply_high_level_action("openEE", [], vel=0.5)

                # post_image
                _pos = [_loc[0], _loc[1], _loc[2] + 3, math.pi, 0, math.pi]
                panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

                # _org_loc_copy[1] = _org_loc_copy[1] - 0.02
                re_arrange_blocks(box_id=_box_id, curr_loc=np.array(_org_loc_copy), sim_handle=panda_handle)

        elif _action_type == "human-move":
            # get the urdf name and remove the existing body
            _obj_name = f"b{_box_id}"
            _obj_id = panda_handle.world.get_obj_id(_obj_name)
            _urdf_name, _, _, _ = panda_handle.world.get_obj_attr(_obj_id)
            pb.removeBody(_obj_id)

            # you have to subtract the table height
            _loc[2] = _loc[2] - panda_handle.world.table_height

            # add a new to the location that human moved-the obj too
            panda_handle.world.load_object(urdf_name=_urdf_name,
                                           obj_name=_obj_name,
                                           obj_init_position=_loc,
                                           obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

        else:
            warnings.warn(f"The current action {_action} does not have a valid action type")
            sys.exit(-1)