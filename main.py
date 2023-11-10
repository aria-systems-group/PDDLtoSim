import os
import re
import time
import random
import datetime
import tracemalloc
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
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import\
    RegretMinimizationStrategySynthesis as RegMinStrSyn
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration

from src.pddl_env_simualtor.envs.panda_sim import PandaSim

from utls import deprecated, timer_decorator

# define a constant to dump the yaml file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def compute_adv_strs(product_graph: TwoPlayerGraph,
                     purely_avd: bool = True,
                     no_intervention: bool = False,
                     cooperative: bool = False,
                     print_sim_str: bool = True) -> List:
    """
    A method to play the adversarial game.
    """
    start = time.time()
    comp_mcr_solver = ValueIteration(product_graph, competitive=True)
    comp_mcr_solver.solve(debug=True, plot=False)
    assert comp_mcr_solver.is_winning() is True, "[Error] There does not exist a winning strategy!"
    stop = time.time()
    print(f"******************************Min-Max Computation time: {stop - start} ****************************")

    # coop_val_dict = coop_mcr_solver.state_value_dict
    comp_str_dict = comp_mcr_solver.str_dict

    _init_state = product_graph.get_initial_states()[0][0]

    _next_state = comp_str_dict[_init_state]
    # print(f"Edge weight: {product_graph.get_edge_weight(_init_state, _next_state)}")
    _action_seq = []

    _action_seq.append(product_graph._graph[_init_state][_next_state][0].get("actions"))
    # print(_action_seq[-1])

    if purely_avd:
        while _next_state is not None:
            _curr_state = _next_state

            _next_state = comp_str_dict.get(_curr_state)

            # print(f"Edge weight: {product_graph.get_edge_weight(_curr_state, _next_state)}")

            if _next_state is not None:
                _edge_act = product_graph._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(product_graph._graph[_curr_state][_next_state][0].get("actions"))
                    # print(_action_seq[-1])

    elif no_intervention:
        while _next_state is not None:
            _curr_state = _next_state

            if product_graph.get_state_w_attribute(_curr_state, "player") == "adam":
                # get the state that sys wanted to evolve to
                for _succ in product_graph._graph.successors(_curr_state):
                    _edge_action = product_graph._graph[_curr_state][_succ][0]["actions"]
                    _edge_type = get_action_from_causal_graph_edge(_edge_action)
                    if _edge_type != "human-move":
                        _next_state = _succ
                        break
            else:
                _next_state = comp_str_dict.get(_curr_state)

            if _next_state is not None:
                _edge_action = product_graph._graph[_curr_state][_next_state][0].get('actions')
                if _action_seq[-1] != _edge_action:
                    _action_seq.append(_edge_action)

    elif cooperative:
        _coop_str_dict = compute_cooperative_actions_for_env(product_graph)
        _max_coop_actions: int = 0
        while _next_state is not None:
            _curr_state = _next_state

            if product_graph.get_state_w_attribute(_curr_state, attribute="player") == "eve":
                _next_state = comp_str_dict.get(_curr_state)
            else:
                if _max_coop_actions <= 2:
                    _next_state = _coop_str_dict[_curr_state]
                    # only increase the counter when the human moves
                    _max_coop_actions += 1
                else:
                    for _succ in product_graph._graph.successors(_curr_state):
                        _edge_action = product_graph._graph[_curr_state][_succ][0]["actions"]
                        _edge_type = get_action_from_causal_graph_edge(_edge_action)
                        if _edge_type != "human-move":
                            _next_state = _succ
                            break

            if _next_state is not None:
                _edge_act = product_graph._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(product_graph._graph[_curr_state][_next_state][0].get("actions"))

    else:
        warnings.warn("Please at-least one of the flags i.e Cooperative, no_intervention or purely_adversarial is True")

    if print_sim_str:
        for _action in _action_seq:
            print(_action)

    return _action_seq


def compute_reg_strs(product_graph: TwoPlayerGraph,
                     coop_str: bool = False,
                     epsilon: float = -1) -> Tuple[list, dict, TwoPlayerGraph]:
    """
    A method to compute strategies. We control the env's behavior by making it purely cooperative, pure adversarial, or
    epsilon greedy.

    @param coop_str: Set this to be true for purely cooperative behavior from the env
    @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks
     random actions.
    """
    # build an instance of regret strategy minimization class
    reg_syn_handle = RegMinStrSyn(product_graph)
    reg_str, reg_val = reg_syn_handle.edge_weighted_arena_finite_reg_solver(reg_factor=1.25,
                                                                            purge_states=True,
                                                                            plot=False)
    twa_game = reg_syn_handle.graph_of_alternatives
    _init_state = twa_game.get_initial_states()[0][0]
    for _n in twa_game._graph.successors(_init_state):
        print(f"Reg Val: {_n}: {reg_val[_n]}")
    # the reg str is dict that one from one state to another. Lets convert this to print a sequence of edge actions
    _next_state = reg_str[_init_state]
    _action_seq = []

    _action_seq.append(twa_game._graph[_init_state][_next_state][0].get("actions"))

    if coop_str:
        # compute cooperative strs for the player
        _coop_str_dict = compute_cooperative_actions_for_env(twa_game)
        _max_coop_actions: int = 1

        # print(f"{_init_state}: {reg_val[_init_state]}")
        # print(f"{_next_state}: {reg_val[_init_state]}")
        while _next_state is not None:
            _curr_state = _next_state

            if twa_game.get_state_w_attribute(_curr_state, attribute="player") == "eve":
                _next_state = reg_str.get(_curr_state)
            else:
                if _max_coop_actions <= 10:
                    _next_state = _coop_str_dict[_curr_state]
                    # only increase the counter when the human moves
                    _max_coop_actions += 1
                else:
                    _next_state = reg_str.get(_curr_state)

            if _next_state is not None:
                _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))
                # print(f"{_next_state}: {reg_val[_init_state]}")

    elif 0 <= epsilon <= 1:
        # we randomise human strategies
        _new_str_dict = compute_epsilon_str_dict(epsilon=epsilon,
                                                 reg_str_dict=reg_str,
                                                 max_human_int=3, twa_game=twa_game)
        while _next_state is not None:
            _curr_state = _next_state

            # if twa_game.get_state_w_attribute(_curr_state, attribute="player") == "eve":
            _next_state = _new_str_dict.get(_curr_state)
            # else:
            #     _new

            if _next_state is not None:
                _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))

    for _action in _action_seq:
        print(_action)

    return _action_seq, reg_val, twa_game


def compute_cooperative_actions_for_env(product_graph: TwoPlayerGraph) -> Dict:
    """
    A helper method to compute the cooperative strategies for the players.
    """
    coop_mcr_solver = ValueIteration(product_graph, competitive=False)
    coop_mcr_solver.cooperative_solver(debug=False, plot=False)
    coop_val_dict = coop_mcr_solver.state_value_dict
    coop_str_dict = coop_mcr_solver.str_dict

    return coop_str_dict


def compute_epsilon_str_dict(epsilon: float, reg_str_dict: dict, max_human_int: int, twa_game: TwoPlayerGraph) -> dict:
    """
    A helper method that return the human action as per the Epsilon greedy algorithm.

    Using this policy we either select a random human action with epsilon probability and the human can select the
    optimal action (as given in the str dict if any) with 1-epsilon probability.

    Epsilon = 0: Env is completely adversarial - Maximizing Sys player's regret
    Epsilon = 1: Env is completely random
    """

    _new_str_dict = reg_str_dict
    if epsilon == 0:
        return _new_str_dict

    _human_has_intervened: int = 0
    for _from_state, _to_state in reg_str_dict.items():
        if twa_game.get_state_w_attribute(_from_state, 'player') == "adam":
            _succ_states: List[tuple] = [_state for _state in twa_game._graph.successors(_from_state)]

            # if human can still intervene
            # if max_human_int >= _human_has_intervened:

                # act random
            if np.random.rand() < epsilon:
                _next_state = random.choice(_succ_states)
                _new_str_dict[_from_state] = _next_state
                # else:
                #     _next_state = _new_str_dict[_from_state]
                #     _human_int_counter = _from_state[0][0][0][1]
                #     if _next_state[0][0][0][1] != _human_int_counter:
                #         _human_has_intervened += 1

            # if human exhausted the limit set by the user
            # else:
            #     _human_int_counter = _from_state[0][0][0][1]
            #     for _succ in _succ_states:
            #         if _succ[0][0][0][1] == _human_int_counter:
            #             _new_str_dict[_from_state] = _succ

    return _new_str_dict


def _manual_rollout(product_graph: TwoPlayerGraph):
    reg_syn_handle = RegMinStrSyn(product_graph)
    reg_str, reg_val = reg_syn_handle.edge_weighted_arena_finite_reg_solver(reg_factor=1.25,
                                                                            purge_states=True,
                                                                            plot=False)
    # print(datetime.datetime.now().time())

    twa_game = reg_syn_handle.graph_of_alternatives

    _init_state = twa_game.get_initial_states()[0][0]
    _action_seq = []

    def _get_successors_based_on_str(__reg_str, __reg_val, __game, __curr_state):
        __succ_list = []
        __succ_state = __reg_str.get(__curr_state, None)

        if __succ_state is None:
            return
        # __val = __reg_val[__succ_state]
        for __count, __n in enumerate(list(__game._graph.successors(__curr_state))):
            # if __reg_val[__n] == __val:
            __edge_action = __game._graph[__curr_state][__n][0].get("actions")
            print(f"[{__count}], state:{__n}, action: {__edge_action}: {reg_val[__n]}")
            __succ_list.append(__n)

        __idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {__succ_list[int(__idx_num)]}")
        return __succ_list[int(__idx_num)]

    _next_state = _get_successors_based_on_str(reg_str, reg_val, twa_game, _init_state)
    _action_seq.append(twa_game._graph[_init_state][_next_state][0].get("actions"))

    while _next_state is not None:
        _curr_state = _next_state

        _next_state = _get_successors_based_on_str(reg_str, reg_val, twa_game, _curr_state)

        if _next_state is not None:
            _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
            if _action_seq[-1] != _edge_act:
                _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))
                print(_action_seq[-1])



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


def save_str(causal_graph: CausalGraph,
             transition_system: FiniteTransitionSystem,
             two_player_game: TwoPlayerGame,
             pos_seq: list,
             regret_graph_of_alternatives: Optional[TwoPlayerGraph] = None,
             game_reg_value: Optional[dict] = None,
             adversarial: bool = False):
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

    if not adversarial:
        _init_state = regret_graph_of_alternatives.get_initial_states()[0][0]
        _reg_value: Optional[int] = game_reg_value.get(_init_state)
    else:
        _reg_value = None

    _init_state = transition_system.transition_system.get_initial_states()[0][0]
    _init_conf = transition_system.transition_system.get_state_w_attribute(_init_state, "list_ap")

    _possible_human_interventions: int = two_player_game.human_interventions

    # transition system nodes and edges
    _trans_sys_nodes = len(transition_system.transition_system._graph.nodes())
    _trans_sys_edges = len(transition_system.transition_system._graph.edges())

    # product graph nodes and edges
    _prod_nodes = len(two_player_game.two_player_game._graph.nodes())
    _prod_edges = len(two_player_game.two_player_game._graph.edges())

    if not adversarial:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = len(regret_graph_of_alternatives._graph.nodes())
        _graph_of_alts_edges = len(regret_graph_of_alternatives._graph.edges())
    else:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = None
        _graph_of_alts_edges = None

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
    if adversarial:
        _file_name: str = \
            f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_{_possible_human_interventions}_h_" \
            f"{_reg_value}_adv_"
    else:
        _file_name: str =\
        f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_{_possible_human_interventions}_h_" \
        f"{_reg_value}_reg_"

    _current_date_time_stamp = str(datetime.datetime.now())

    # remove the seconds stamp
    _time_stamp, *_ = _current_date_time_stamp.partition('.')
    _time_stamp = _time_stamp.replace(" ", "_" )
    _time_stamp = _time_stamp.replace(":", "_")
    _time_stamp = _time_stamp.replace("-", "_")
    _file_path = ROOT_PATH + _file_name + _time_stamp + ".yaml"
    try:
        with open(_file_path, 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False, sort_keys=False)

    except FileNotFoundError:
        print(FileNotFoundError)
        print(f"The file {_file_path} could not be found."
              f" This could be because I could not find the folder to dump in")


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


def load_data_from_yaml_file(file_add: str) -> Dict:
    """
    A helper function to load the sequence of strategies given a valid yaml file.
    """

    try:
        with open(file_add, 'r') as stream:
            graph_data = yaml.load(stream, Loader=yaml.Loader)

    except FileNotFoundError as error:
        print(error)
        print(f"The file does not exist at the loc {file_add}")

    return graph_data

@timer_decorator
def daig_main(print_flag: bool = False, record_flag: bool = False) -> None:
    _project_root = os.path.dirname(os.path.abspath(__file__))

    _domain_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    # _problem_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/problem.pddl"
    _problem_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/sym_test_problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")
    start = time.time()
    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    # _transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.edges())}")

    _two_player_instance = TwoPlayerGame(_causal_graph_instance, _transition_system_instance)
    _two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=False)

    # product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)

    # for implicit construction, the human intervention should >=2
    _two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    _two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    _two_player_instance.modify_ap_w_object_types(implicit=True)
    _two_player_instance.modify_edge_weights(implicit=True)
    stop = time.time()
    print(f"******************************Original Graph construction time: {stop - start}******************************")

    # print # of Sys and Env state
    env_count = 0
    sys_count = 0
    for (p, d) in _two_player_instance._two_player_implicit_game._graph.nodes(data=True):
        if d['player'] == 'adam':
            env_count += 1
        elif d['player'] == 'eve':
            sys_count += 1

    print(f"# of Sys states in Two player game: {sys_count}")
    print(f"# of Env states in Two player game: {env_count}")

    # for (u, v) in _two_player_instance._two_player_implicit_game._graph.edges():
    #     print(f"{u} -------{_two_player_instance._two_player_implicit_game._graph[u][v][0].get('actions')}------> {v}")
    # sys.exit(-1)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.edges())}")

    # _dfa = _two_player_instance.build_LTL_automaton(formula="F(l2 || l6)")
    # _dfa = _two_player_instance.build_LTL_automaton(
    #     formula="F((p22 & p14 & p03) || (p05 & p19 & p26))")
    _dfa = _two_player_instance.build_LTL_automaton(
        # formula="F((p03 & p14 & p22) || (p05 & p19 & p26))")
        formula="F(p01 || p17)")
    # _dfa = _two_player_instance.build_LTL_automaton(
    #     formula="F((p12 & p00) || (p20 & p12) || (p05 & p19) || (p25 & p19))")

    _product_graph = _two_player_instance.build_product(dfa=_dfa,
                                                        trans_sys=_two_player_instance.two_player_implicit_game)
    _relabelled_graph = _two_player_instance.internal_node_mapping(_product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(_relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(_relabelled_graph._graph.edges())}")

    # compute strs
    # _actions, _reg_val, _graph_of_alts = compute_reg_strs(_product_graph, coop_str=False, epsilon=0)
    # _manual_rollout(product_graph=_product_graph)

    # adversarial strs
    _actions = compute_adv_strs(_product_graph,
                                purely_avd=True,
                                no_intervention=False,
                                cooperative=False,
                                print_sim_str=True)

    # ask the user if they want to save the str or not
    # _dump_strs = input("Do you want to save the strategy,Enter: Y/y")
    _dump_strs = "n"
    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=_causal_graph_instance,
                 transition_system=_transition_system_instance,
                 two_player_game=_two_player_instance,
                 regret_graph_of_alternatives=_graph_of_alts,
                 game_reg_value=_reg_val,
                 pos_seq=_actions,
                 adversarial=False)

    # simulate the str
    # execute_str(actions=_actions,
    #             causal_graph=_causal_graph_instance,
    #             transition_system=_transition_system_instance,
    #             exp_name="diag",
    #             record_sim=record_flag,
    #             debug=False)


def arch_main(print_flag: bool = False, record_flag: bool = False) -> None:
    _project_root = os.path.dirname(os.path.abspath(__file__))

    _domain_file_path = _project_root + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = _project_root + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")

    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    _transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)
    _transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.edges())}")

    _two_player_instance = TwoPlayerGame(_causal_graph_instance, _transition_system_instance)
    _two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=True)

    # for implicit construction, the human intervention should >=2
    _two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    _two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    # two_player_instance.modify_ap_w_object_types(implicit=True)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.edges())}")

    _dfa = _two_player_instance.build_LTL_automaton(formula="F((l8 & l9 & l0) || (l3 & l2 & l1))")
    # _dfa = _two_player_instance.build_LTL_automaton(formula="F(l8 & l9 & l0)")
    # _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                                     trans_sys=_two_player_instance.two_player_game)

    _product_graph = _two_player_instance.build_product(dfa=_dfa,
                                                        trans_sys=_two_player_instance.two_player_implicit_game)

    _relabelled_graph = _two_player_instance.internal_node_mapping(_product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(_relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(_relabelled_graph._graph.edges())}")

    # compute strs
    _actions, _reg_val, _graph_of_alts = compute_reg_strs(_product_graph, coop_str=False, epsilon=0)

    # adversarial strs
    # _actions = compute_adv_strs(_product_graph,
    #                             purely_avd=True,
    #                             no_intervention=False,
    #                             cooperative=False,
    #                             print_sim_str=True)
    exit()

    # ask the user if they want to save the str or not
    _dump_strs = input("Do you want to save the strategy,Enter: Y/y")

    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=_causal_graph_instance,
                 transition_system=_transition_system_instance,
                 two_player_game=_two_player_instance,
                 # regret_graph_of_alternatives=_graph_of_alts,
                 # game_reg_value=_reg_val,
                 pos_seq=_actions,
                 adversarial=True)

    # simulate the str
    execute_str(actions=_actions,
                causal_graph=_causal_graph_instance,
                transition_system=_transition_system_instance,
                exp_name="arch",
                record_sim=record_flag,
                debug=False)


if __name__ == "__main__":
    record = False
    use_saved_str = False

    if use_saved_str:
        # get the actions from the yaml file
        file_name = "/arch_2_tables_4_box_6_loc_2_h_2_reg_2021_04_28_14_23_52.yaml"
        file_pth: str = ROOT_PATH + "/saved_strs" + file_name

        yaml_dump = load_data_from_yaml_file(file_add=file_pth)

        execute_saved_str(yaml_data=yaml_dump,
                          exp_name="arch",
                          record_sim=record,
                          debug=False)

    else:
        # starting the monitoring
        tracemalloc.start()
        daig_main(print_flag=True, record_flag=record)
        # arch_main(print_flag=False, record_flag=record)

        # displaying the memory - output current memory usage and peak memory usage
        _,  peak_mem = tracemalloc.get_traced_memory()
        print(f" Peak memory [MB]: {peak_mem/(1024*1024)}")
        # stopping the library
        tracemalloc.stop()