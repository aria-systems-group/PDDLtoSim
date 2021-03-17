import os
import time
import yaml
import copy
import sys
import math
import warnings
import pybullet as pb
import numpy as np

from typing import Tuple, Optional, List, Dict

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

# call the regret synthesis code
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
from regret_synthesis_toolbox.src.payoff import payoff_factory
from regret_synthesis_toolbox.src.strategy_synthesis import RegMinStrSyn

from src.pddl_env_simualtor.envs.panda_sim import PandaSim

# define a constant to dump the yaml file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def compute_reg_strs(product_graph: TwoPlayerGame) -> Tuple[list, np.int32, TwoPlayerGraph]:
    # _init_state = product_graph.get_initial_states()[0][0]

    payoff = payoff_factory.get("cumulative", graph=product_graph)

    # build an instance of regret strategy minimization class
    reg_syn_handle = RegMinStrSyn(product_graph, payoff)
    reg_str, reg_val = reg_syn_handle.edge_weighted_arena_finite_reg_solver(minigrid_instance=None,
                                                                            purge_states=True,
                                                                            plot=False)
    twa_game = reg_syn_handle.graph_of_alternatives
    _init_state = twa_game.get_initial_states()[0][0]
    # the reg str is dict that one from one state to another. Lets convert this to print a sequence of edge actions
    _next_state = reg_str[_init_state]
    _action_seq = []

    _action_seq.append(twa_game._graph[_init_state][_next_state][0].get("actions"))

    print(_init_state)
    print(_next_state)
    while _next_state is not None:
        _curr_state = _next_state
        _next_state = reg_str.get(_curr_state)
        if _next_state is not None:
            _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
            if _action_seq[-1] != _edge_act:
                _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))
            print(_next_state)

    for _action in _action_seq:
        print(_action)

    return _action_seq, reg_val, twa_game


def execture_str(actions: list,
                 causal_graph: CausalGraph,
                 transition_system: FiniteTransitionSystem,
                 record_sim: bool = False,
                 debug: bool = False):
    # determine the action type first
    _action_type = ""
    _loc_dict = load_pre_built_loc_info("diag")

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

    for _action in actions:
        _action_type = transition_system._get_action_from_causal_graph_edge(_action)
        _box_id, _loc = transition_system._get_multiple_box_location(_action)
        if len(_loc) == 2:
            _from_loc = _loc[0]
            _to_loc = _loc[1]
        else:
            _to_loc = _loc[0]
        _loc = _loc_dict.get(_to_loc)

        if _action_type == "transit":
            # pre-image based on the object loc
            if _loc[0] < 0:
                panda_handle.apply_high_level_action("transit", _wait_pos_left, vel=0.5)
            else:
                panda_handle.apply_high_level_action("transit", _wait_pos_right, vel=0.5)
            # every transfer and transit action will have a from and to location. Lets extract it.
            _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
            # panda_handle.apply_high_level_action("openEE", [])
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

        elif _action_type == "transfer":
            # pre-image
            # if _loc[0] < 0:
            #     panda_handle.apply_high_level_action("transfer", _wait_pos_left, vel=0.5)
            # else:
            #     panda_handle.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

            _pos = [_loc[0], _loc[1], _loc[2] + 0.3, math.pi, 0, math.pi]
            # panda_handle.apply_high_level_action("openEE", [])
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
            # pre-image
            _pos = [_loc[0], _loc[1], _loc[2] + 0.05, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transfer", _pos, vel=0.5)

            panda_handle.apply_high_level_action("openEE", [], vel=0.5)

            #post_image
            _pos = [_loc[0], _loc[1], _loc[2] + 3, math.pi, 0, math.pi]
            panda_handle.apply_high_level_action("transit", _pos, vel=0.5)

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


def initialize_simulation(causal_graph: CausalGraph,
                          transition_system: FiniteTransitionSystem,
                          loc_dict: dict,
                          record_sim: bool = False,
                          debug: bool = False):
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

    # intialize objects at the corresponding locs
    _init_state = transition_system.transition_system.get_initial_states()[0][0]
    _init_conf = transition_system.transition_system.get_state_w_attribute(_init_state, "list_ap")

    for _idx, _loc in enumerate(_init_conf):
        if _idx == len(_init_conf) - 1:
            continue
        panda.world.load_object(urdf_name="red_box",
                                obj_name=f"b{_idx}",
                                obj_init_position=copy.copy(loc_dict.get(_loc)),
                                obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

    for _loc in box_locs:
        _visual_marker_loc = copy.copy(loc_dict.get(_loc))
        _visual_marker_loc[2] = 0
        panda.world.load_markers(marker_loc=_visual_marker_loc)

    return panda


# def _pre_loaded_pick_and_place_action(pos):
#     _wait_pos_left = [-0.2, 0.0, 0.9, math.pi, 0, math.pi]
#     _wait_pos_right = [0.2, 0.0, 0.9, math.pi, 0, math.pi]
#
#     # give pose and orientation
#     panda.apply_high_level_action("openEE", [])
#
#     panda.apply_high_level_action("transit", _wait_pos_left, vel=0.5)
#
#     _pos = [pos[0], pos[1], pos[2] + 0.3, math.pi, 0, math.pi]
#     panda.apply_high_level_action("transit", _pos, vel=0.5)
#
#     # # 2. go down towards the object
#     _pos = [pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
#     panda.apply_high_level_action("transit", _pos, vel=0.5)
#
#     panda.apply_high_level_action("closeEE", [], vel=0.5)
#
#     # # 3. grab the object
#     _pos = [pos[0], pos[1], pos[2] + 0.3,  math.pi, 0, math.pi]
#     panda.apply_high_level_action("transfer", _pos, vel=0.5)
#
#     panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)
#
#     _pos = [-pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
#     panda.apply_high_level_action("transfer", _pos, vel=0.5)
#
#     panda.apply_high_level_action("openEE", [], vel=0.5)

def save_str(causal_graph: CausalGraph,
             transition_system: FiniteTransitionSystem,
             two_player_game: TwoPlayerGame,
             regret_graph_of_alternatives: TwoPlayerGraph,
             game_reg_value: np.int32,
             pos_seq: list):
    """
    A helper method that dumps the regret value and the corresponding strategy computed for given abstraction and an
    LTL formula. This method creates a yaml file which is then dumped in the saved_strs folder at the root of the
    project. The file naming convention is
    <task_name>_<# of boxes>_<# of locs>_<# of possible human intervention>_<reg_value>.

    The stuff being dumped is :
        1. Task Name
        2. # of boxes along with the names
        3. # of locs along with the names
        4. # of possible human interventions
        5. # of nodes in the game
        6. # of edges in the game
        7. LTL Formula used
        8. strategy compute (a sequence of actions)
    """
    
    _task_name: str = causal_graph.get_task_name()
    _boxes = causal_graph.task_objects
    _locations = causal_graph.task_locations

    _reg_value: int = game_reg_value.item()

    _init_state = transition_system.transition_system.get_initial_states()[0][0]
    _init_conf = transition_system.transition_system.get_state_w_attribute(_init_state, "list_ap")

    _possible_human_interventions: int = two_player_game.human_interventions

    # transition system nodes and edges
    _trans_sys_nodes = len(transition_system.transition_system._graph.nodes())
    _trans_sys_edges = len(transition_system.transition_system._graph.edges())

    # product graph nodes and edges
    _prod_nodes = len(two_player_game.two_player_game._graph.nodes())
    _prod_edges = len(two_player_game.two_player_game._graph.edges())

    # graph of alternatives nodes and edges
    _graph_of_alts_nodes = len(regret_graph_of_alternatives._graph.nodes())
    _graph_of_alts_edges = len(regret_graph_of_alternatives._graph.edges())

    _ltl_formula = two_player_game.formula

    # create a data dict to dump it
    data_dict: Dict = dict(
        task_name=_task_name,
        no_of_boxes={
            'num': len(_boxes),
            'objects': _boxes,
        },
        no_of_loc={
            'num': len(_locations),
            'objects': _locations
        },
        max_human_int=_possible_human_interventions,
        init_worl_conf=_init_conf,
        ltl_formula=_ltl_formula,
        abstractions={
            'num_transition_system_nodes': _trans_sys_nodes,
            'num_transition_system_edges': _trans_sys_edges,
            'num_two_player_game_nodes': _prod_nodes,
            'num_two_player_game_edges': _prod_edges,
            'num_graph_of_alts_nodes': _graph_of_alts_nodes,
            'num_graph_of_alts_edges': _graph_of_alts_edges,
        },
        reg_val=_reg_value,
        reg_str=pos_seq
    )

    # now dump the data in a file
    _file_name: str =\
    f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_{_possible_human_interventions}_h_{_reg_value}_reg.yaml"

    _file_path = ROOT_PATH + _file_name
    try:
        with open(_file_path, 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False, sort_keys=False)

    except FileNotFoundError:
        print(FileNotFoundError)
        print(f"The file {_file_path} could not be found."
              f" This could be because I could not find the folder to dump in")


def load_pre_built_loc_info(exp_name: str):
    if exp_name == "diag":
        loc_dict = {
            # 'l0': np.array([-0.7, -0.2, 0.17/2]),
            # 'l2': np.array([-0.4, 0.2, 0.17/2]),
            # 'l1': np.array([-0.4, -0.2, 0.17/2]),
            # 'l3': np.array([0.6, -0.2, 0.17/2]),
            # 'l4': np.array([0.45, 0.0, 0.17/2])
            'l0': np.array([-0.6, -0.2, 0.17 / 2]),
            'l2': np.array([-0.6, 0.2, 0.17 / 2]),
            'l3': np.array([0.4, -0.2, 0.17 / 2]),
            'l1': np.array([-0.4, 0.2, 0.17 / 2]),
            'l4': np.array([0.5, 0.0, 0.17 / 2])
        }
    elif exp_name == "arch":
        pass
    else:
        warnings.warn("PLease enter a valid experiment name")

    return loc_dict

def load_data_from_yaml_file(file_add: str) -> Dict:
    """
    A helper function to load the sequence of strategies given a valid yaml file.
    """

    try:
        with open(file_add, 'r') as stream:
            graph_data = yaml.load(stream, Loader=yaml.Loader)

    except FileNotFoundError as error:
        print(error)
        print(f"The file {file_name} does not exist")

    return graph_data


if __name__ == "__main__":
    record = False
    dump_strs = True
    use_saved_str = False

    # build the product automaton
    _project_root = os.path.dirname(os.path.abspath(__file__))

    # Experimental stage - lets try calling the function within pyperplan
    domain_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    problem_file_ath = _project_root + "/pddl_files/two_table_scenario/diagonal/problem.pddl"

    causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=False)

    causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)
    print(
        f"No. of nodes in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")
    print(
        f"No. of edges in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")

    transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    transition_system_instance.build_transition_system(plot=False)
    transition_system_instance.modify_edge_weights()

    print(f"No. of nodes in the Transition System is :{len(transition_system_instance.transition_system._graph.nodes())}")
    print(f"No. of edges in the Transition System is :{len(transition_system_instance.transition_system._graph.nodes())}")

    two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    two_player_instance.build_two_player_game(human_intervention=2, plot_two_player_game=False)
    two_player_instance.set_appropriate_ap_attribute_name()
    two_player_instance.modify_ap_w_object_types()

    dfa = two_player_instance.build_LTL_automaton(formula="F((p01 & p12) || (p03 & p14))")
    product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)
    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)
    # relabelled_graph.plot_graph()

    # print some details about the product graph
    print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    if not use_saved_str:
        # compute strs
        actions, reg_val, graph_of_alts = compute_reg_strs(product_graph)

        # save strs
        if dump_strs:
            save_str(causal_graph=causal_graph_instance,
                     transition_system=transition_system_instance,
                     two_player_game=two_player_instance,
                     regret_graph_of_alternatives=graph_of_alts,
                     game_reg_value=reg_val,
                     pos_seq=actions)

        # simulate the str
        execture_str(actions=actions,
                     causal_graph=causal_graph_instance,
                     transition_system=transition_system_instance,
                     record_sim=record,
                     debug=False)
    else:

        # get the actions from the yaml file
        file_name = "/5loc_problem_1_box_2_loc_2_h_0_reg.yaml"
        file_pth: str = ROOT_PATH + "/saved_strs" + file_name

        yaml_dump = load_data_from_yaml_file(file_add=file_pth)
        actions = yaml_dump.get("reg_str")

        # TODO: Write a dedicated method to load all the data relevant to simulation from the yaml file alone.
        execture_str(actions=actions,
                     causal_graph=causal_graph_instance,
                     transition_system=transition_system_instance,
                     record_sim=record,
                     debug=False)

    # build the simulator
    # if record:
    #     physics_client = pb.connect(pb.GUI,
    #                                 options="--minGraphicsUpdateTimeMs=0 --mp4=\"experiment.mp4\" --mp4fps=240")
    # else:
    #     physics_client = pb.connect(pb.GUI)
    #
    # panda = PandaSim(physics_client, use_IK=1)
    # # _loc_dict = load_pre_built_loc_info("diag")
    # _loc_dict = {
    #     # 0: np.array([-0.5, -0.2, 0.17 / 2]),
    #     2: np.array([-0.5, 0.2, 0.17 / 2]),
    #     3: np.array([-0.3, -0.2, 0.17 / 2]),
    #     1: np.array([-0.3, 0.2, 0.17 / 2]),
    #     4: np.array([-0.4, 0.0, 0.17 / 2])
    # }
    # _obj_loc = []
    # for i in range(1):
    #     _obj_loc.append(_loc_dict.get(2))
    #     panda.world.load_object(urdf_name="red_box",
    #                             obj_name=f"red_box_{2}",
    #                             obj_init_position=_loc_dict.get(2),
    #                             obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))

    #     _loc = copy.copy(_loc_dict.get(i))
    #
    #     _loc[2] = 0.625
    #
    #     visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_BOX,
    #                          halfExtents=[0.05, 0.05, 0.001],
    #                          rgbaColor=[0, 0, 1, 0.6],
    #                          specularColor=[0.4, .4, 0],
    #                          visualFramePosition=_loc/2,
    #                          physicsClientId=panda._physics_client_id)
    #
    #     collision_shape_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX,
    #                                                  halfExtents=[0.05, 0.05, 0.001],
    #                                                  collisionFramePosition=_loc/2,
    #                                                  physicsClientId=panda._physics_client_id)
    #
    #     pb.createMultiBody(baseMass=0,
    #                        # baseInertialFramePosition=[0, 0, 0],
    #                        # baseCollisionShapeIndex=collision_shape_id,
    #                        baseVisualShapeIndex=visual_shape_id,
    #                        basePosition=_loc/2,
    #                        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
    #                        physicsClientId=panda._physics_client_id)
    #
    # for i in range(3):
    #     _obj_loc.append(_loc_dict.get(i))
    #     _loc = copy.copy(_loc_dict.get(i))
    #
    #     _loc[2] = 0.625
    #     _loc[0] = -1 * _loc[0]
    #
    #     visual_shape_id = pb.createVisualShape(shapeType=pb.GEOM_BOX,
    #                                            halfExtents=[0.05, 0.05, 0.001],
    #                                            rgbaColor=[0, 0, 1, 0.6],
    #                                            specularColor=[0.4, .4, 0],
    #                                            visualFramePosition=_loc / 2,
    #                                            physicsClientId=panda._physics_client_id)
    #
    #     collision_shape_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX,
    #                                                  halfExtents=[0.05, 0.05, 0.001],
    #                                                  collisionFramePosition=_loc / 2,
    #                                                  physicsClientId=panda._physics_client_id)
    #
    #     pb.createMultiBody(baseMass=0,
    #                        # baseInertialFramePosition=[0, 0, 0],
    #                        # baseCollisionShapeIndex=collision_shape_id,
    #                        baseVisualShapeIndex=visual_shape_id,
    #                        basePosition=_loc / 2,
    #                        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
    #                        physicsClientId=panda._physics_client_id)


    # for _ in range(1500):
    # for i, loc in enumerate(_obj_loc):
    #     # if i != 0:
    #     #     panda.apply_action(panda._home_hand_pose)
    #     #     for _ in range(100):
    #     #         pb.stepSimulation()
    #     #         time.sleep(0.01)
    #     # for _ in range(2):
    #     _pre_loaded_pick_and_place_action(loc)