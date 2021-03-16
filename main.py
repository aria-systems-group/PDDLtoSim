import os
import time
import copy
import sys
import math
import warnings
import pybullet as pb
import numpy as np

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

# call the regret synthesis code
from regret_synthesis_toolbox.src.payoff import payoff_factory
from regret_synthesis_toolbox.src.strategy_synthesis import RegMinStrSyn

from src.pddl_env_simualtor.envs.panda_sim import PandaSim


def compute_reg_strs(product_graph: TwoPlayerGame) -> list:
    # _init_state = product_graph.get_initial_states()[0][0]

    payoff = payoff_factory.get("cumulative", graph=product_graph)

    # build an instance of regret strategy minimization class
    reg_syn_handle = RegMinStrSyn(product_graph, payoff)
    reg_str = reg_syn_handle.edge_weighted_arena_finite_reg_solver(minigrid_instance=None,
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

    return _action_seq


def execture_str(actions: list,
                 causal_graph: CausalGraph,
                 transition_system: FiniteTransitionSystem):
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
                                         record_sim=True,
                                         debug=False)

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
    return panda



def _pre_loaded_pick_and_place_action(pos):
    _wait_pos_left = [-0.2, 0.0, 0.9, math.pi, 0, math.pi]
    _wait_pos_right = [0.2, 0.0, 0.9, math.pi, 0, math.pi]

    # give pose and orientation
    panda.apply_high_level_action("openEE", [])

    panda.apply_high_level_action("transit", _wait_pos_left, vel=0.5)

    # panda.apply_action(panda._home_hand_pose)
    # # panda.apply_action(action=_pos, max_vel=0.5)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)

    _pos = [pos[0], pos[1], pos[2] + 0.3, math.pi, 0, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=2)
    # panda.pre_grasp(max_velocity=2)
    #
    # for _ in range(800):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 2. go down towards the object
    _pos = [pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
    panda.apply_high_level_action("transit", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=0.5)
    #
    # for _ in range(600):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # panda.grasp(obj_id=3)
    panda.apply_high_level_action("closeEE", [], vel=0.5)
    # for _ in range(300):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 3. grab the object
    _pos = [pos[0], pos[1], pos[2] + 0.3,  math.pi, 0, math.pi]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)
    # panda.apply_action(action=_pos, max_vel=0.5)
    # panda.grasp(obj_id=3)
    #
    # for _ in range(400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    panda.apply_high_level_action("transfer", _wait_pos_right, vel=0.5)

    _pos = [-pos[0], pos[1], pos[2] + 0.05, math.pi, 0, math.pi]
    panda.apply_high_level_action("transfer", _pos, vel=0.5)

    panda.apply_high_level_action("openEE", [], vel=0.5)

    # _pre_pose = [0.2, 0.0, 0.9,
    #             min(math.pi, max(-math.pi, math.pi)),
    #             min(math.pi, max(-math.pi, 0)),
    #             min(math.pi, max(-math.pi, 0))]
    #
    # panda.apply_action(_pre_pose)
    # panda.grasp(obj_id=3)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # 4. pick and place it somewhere else
    # _pos = [-pos[0], pos[1], pos[2] + 0.05]
    # panda.apply_action(action=_pos, max_vel=0.5)
    # panda.grasp(obj_id=3)
    #
    # for _ in range(800):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # # release
    # panda.pre_grasp()
    # for _ in range(300):
    #     pb.stepSimulation()
    #     time.sleep(0.01)
    #
    # panda.apply_action(panda._home_hand_pose)
    # # panda.apply_action(action=_pos, max_vel=0.5)
    # for _ in range(1400):
    #     pb.stepSimulation()
    #     time.sleep(0.01)


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
            'l3': np.array([-0.4, -0.2, 0.17 / 2]),
            'l1': np.array([-0.4, 0.2, 0.17 / 2]),
            'l4': np.array([-0.5, 0.0, 0.17 / 2])
        }
    elif exp_name == "arch":
        pass
    else:
        warnings.warn("PLease enter a valid experiment name")

    return loc_dict

if __name__ == "__main__":
    record = False

    # build the product automaton
    _project_root = os.path.dirname(os.path.abspath(__file__))

    # Experimental stage - lets try calling the function within pyperplan
    domain_file_path = _project_root + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/pddl_files/blocks_world/problem.pddl"

    causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=False)

    causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)
    print(
        f"No. of nodes in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")
    print(
        f"No. of edges in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")

    transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    transition_system_instance.build_transition_system(plot=False)

    print(f"No. of nodes in the Transition System is :{len(transition_system_instance.transition_system._graph.nodes())}")
    print(f"No. of edges in the Transition System is :{len(transition_system_instance.transition_system._graph.nodes())}")

    two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    two_player_instance.build_two_player_game(human_intervention=2, plot_two_player_game=False)
    two_player_instance.set_appropriate_ap_attribute_name()
    # two_player_instance.modify_ap_w_object_types()

    dfa = two_player_instance.build_LTL_automaton(formula="F(l2 & l4 & l3)")
    product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)
    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)
    # relabelled_graph.plot_graph()

    # print some details about the product graph
    print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    # compute strs
    actions = compute_reg_strs(product_graph)

    # simulate the str
    execture_str(actions=actions,
                 causal_graph=causal_graph_instance,
                 transition_system=transition_system_instance)

    # build the simulator
    # if record:
    #     physics_client = pb.connect(pb.GUI,
    #                                 options="--minGraphicsUpdateTimeMs=0 --mp4=\"experiment.mp4\" --mp4fps=240")
    # else:
    #     physics_client = pb.connect(pb.GUI)
    #
    # panda = PandaSim(physics_client, use_IK=1)
    # _loc_dict = load_pre_built_loc_info("diag")
    # _obj_loc = []
    # for i in range(3):
    #     _obj_loc.append(_loc_dict.get(i))
    #     panda.world.load_object(urdf_name="red_box",
    #                             obj_name=f"red_box_{i}",
    #                             obj_init_position=_loc_dict.get(i),
    #                             obj_init_orientation=pb.getQuaternionFromEuler([0, 0, 0]))
    #
    # # for _ in range(1500):
    # for i, loc in enumerate(_obj_loc):
    #     # if i != 0:
    #     #     panda.apply_action(panda._home_hand_pose)
    #     #     for _ in range(100):
    #     #         pb.stepSimulation()
    #     #         time.sleep(0.01)
    #     # for _ in range(2):
    #     _pre_loaded_pick_and_place_action(loc)