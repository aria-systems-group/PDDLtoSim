import re
import warnings
import copy
import networkx as nx

from typing import Tuple, Dict, List, Optional
from collections import deque, defaultdict
from bidict import bidict

from regret_synthesis_toolbox.src.graph import graph_factory
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

# import local packages
from .causal_graph import CausalGraph
from .transition_system import FiniteTransitionSystem


class TwoPlayerGame:
    """
    A Class that builds a Two player game based on the Transition System built using the Causal Graph.
    """

    def __init__(self, causal_graph, transition_system):
        self._causal_graph: CausalGraph = causal_graph
        self._transition_system: FiniteTransitionSystem = transition_system
        self._two_player_game: Optional[TwoPlayerGraph] = None
        self._two_player_implicit_game: Optional[TwoPlayerGraph] = None

    @property
    def causal_graph(self):
        return self._causal_graph

    @property
    def transition_system(self):
        return self._transition_system

    @property
    def two_player_game(self):
        if isinstance(self._two_player_game, type(None)):
            warnings.warn("The Two player game is of type of None. Please build the game before accessing it")
        return self._two_player_game

    @property
    def two_player_implicit_game(self):
        if isinstance(self._two_player_game, type(None)):
            warnings.warn("The Two player implicit game is of type of None. Please build the explicit game before"
                          "accessing it")
        return self._two_player_implicit_game

    @property
    def human_interventions(self):
        return self._human_interventions

    @property
    def formula(self):
        return self._formula

    def build_two_player_game(self,
                              human_intervention: int = 1,
                              human_intervention_cost: int = 0,
                              human_non_intervention_cost: int = 0,
                              plot_two_player_game: bool = False,
                              relabel_nodes: bool = True,
                              arch_construction: bool = False):
        """
        A function that build a Two Player Game based on a Transition System built from causal graph.

        After every Sys transition add a Human state, we then add valid human transitions from that human state. Every
        node that belongs to the Two player game has the same node as well as the edge attributes as their counterparts
        in the Transition System.
        """
        self._human_interventions = human_intervention
        _init_state = self._transition_system.transition_system.get_initial_states()[0][0]

        eve_node_lst = []
        adam_node_lst = []
        _two_plr_to_sgl_plr_sys_mapping: Dict[Tuple, dict] = defaultdict(lambda: {})
        _graph_name = "two_player" + self._causal_graph.task.name
        _config_yaml = "/config/" + "two_player" + self._causal_graph.task.name

        self._two_player_game = graph_factory.get("TwoPlayerGraph",
                                                  graph_name=_graph_name,
                                                  config_yaml=_config_yaml,
                                                  save_flag=True,
                                                  pre_built=False,
                                                  plot=False)

        # lets create k copies of the system states
        for _n in self._transition_system.transition_system._graph.nodes():
            for i in range(human_intervention + 1):
                _sys_node = (_n, i)
                _two_plr_to_sgl_plr_sys_mapping[_sys_node] = self._transition_system.transition_system._graph.nodes[_n]

                if _sys_node in eve_node_lst:
                    warnings.warn("The Transition System seems to contain multiple states of same configuration."
                                  "Check your Causal graph construction and Transition system construction functions")
                else:
                    eve_node_lst.append(_sys_node)

        for _u in eve_node_lst:
            if not self._two_player_game._graph.has_node(_u):
                # add all the attributes from the single player TS to this two-player sys node
                _single_player_sys_node = _two_plr_to_sgl_plr_sys_mapping.get(_u)

                if _single_player_sys_node is not None:
                    self._two_player_game.add_state(_u, **_single_player_sys_node)
                    self._two_player_game._graph.nodes[_u]['init'] = False

                    if _u[0] == _init_state and _u[1] == human_intervention:
                        self._two_player_game._graph.nodes[_u]['init'] = True

        # for each edge create a human node and then add valid human transition from that human state
        for _e in self._transition_system.transition_system._graph.edges():
            for i in reversed(range(human_intervention + 1)):
                _u = _e[0]
                _v = _e[1]
                _edge_action = self._transition_system.transition_system._graph.get_edge_data(*_e)[0]['actions']

                _env_node = (f"h{_u}{_edge_action}", i)
                adam_node_lst.append(_env_node)

                # add this node to the game and the attributes of the sys state.
                # Change player and causal state attribute to "adam" and "human-move" respectively.
                if not self._two_player_game._graph.has_node(_env_node):
                    _single_player_sys_node = _two_plr_to_sgl_plr_sys_mapping.get((_u, i))

                    if _single_player_sys_node is not None:
                        self._two_player_game.add_state(_env_node, **_single_player_sys_node)
                        self._two_player_game._graph.nodes[_env_node]['player'] = "adam"
                        self._two_player_game._graph.nodes[_env_node]['causal_state_name'] = "human-move"
                        self._two_player_game._graph.nodes[_env_node]['init'] = False
                else:
                    warnings.warn(
                        f"The human state {_env_node} already exists. This is a major blunder in the code")

                # get the org edge and its attributes between _u and _v
                _org_edge_attributes = self._transition_system.transition_system._graph.edges[_u, _v, 0]

                # add edge between the original system state and the human state
                self._two_player_game.add_edge(u=(_u, i),
                                               v=_env_node,
                                               **_org_edge_attributes)

                # add a valid human nonintervention edge and its corresponding action cost
                self._two_player_game.add_edge(u=_env_node,
                                               v=(_v, i),
                                               **_org_edge_attributes)
                self._two_player_game._graph.edges[_env_node, (_v, i), 0]['weight'] = human_non_intervention_cost

                if i != 0:
                    # now get add all the valid human interventions
                    self._add_valid_human_edges(human_state_name=_env_node,
                                                org_succ_state_name=(_v, i),
                                                human_intervention_cost=human_intervention_cost,
                                                arch_construction=arch_construction)

        self.__add_transition_from_new_sys_states(print_new_states=False, arch_construction=arch_construction)

        # after adding valid transitions from novel Sys states to existing Sys states. We need to once again add human
        # state associated with these edges.
        _old_two_player_pddl_ts: TwoPlayerGraph = copy.deepcopy(self._two_player_game)

        for _e in _old_two_player_pddl_ts._graph.edges():
            _u = _e[0]
            _v = _e[1]
            i: int = _u[1]

            if self._two_player_game.get_state_w_attribute(_u, "player") == "adam" or\
                    self._two_player_game.get_state_w_attribute(_v, "player") == "adam":
                continue

            _edge_action = self._two_player_game._graph.get_edge_data(*_e)[0]['actions']

            _env_node = (f"h{_u}{_edge_action}", i)
            adam_node_lst.append(_env_node)

            if not self._two_player_game._graph.has_node(_env_node):
                _sys_node_attrs = self._two_player_game._graph.nodes[_u]
                self._two_player_game.add_state(_env_node, **_sys_node_attrs)
                self._two_player_game._graph.nodes[_env_node]['player'] = "adam"
                self._two_player_game._graph.nodes[_env_node]['causal_state_name'] = "human-move"

            else:
                warnings.warn(f"The human state {_env_node} already exists. This is a major blunder in the code")

            # get the org edge and its attributes between _u and _v
            _org_edge_attributes = self._two_player_game._graph.edges[_u, _v, 0]

            # add edge between the original system state and the human state
            self._two_player_game.add_edge(u=_u,
                                           v=_env_node,
                                           **_org_edge_attributes)

            # add a valid human nonintervention edge and its corresponding action cost
            self._two_player_game.add_edge(u=_env_node,
                                           v=_v,
                                           **_org_edge_attributes)
            self._two_player_game._graph.edges[_env_node, _v, 0]['weight'] = human_non_intervention_cost

            # remove the original _u to _v edge
            self._two_player_game._graph.remove_edge(_u, _v)

            if i != 0:
                # now get add all the valid human interventions
                self._add_valid_human_edges(human_state_name=_env_node,
                                            org_succ_state_name=_v,
                                            human_intervention_cost=human_intervention_cost,
                                            arch_construction=arch_construction)

        print("Iterating for the second time to check if human interventions created any new nodes")
        self.__add_transition_from_new_sys_states(print_new_states=False, arch_construction=arch_construction)

        if plot_two_player_game:
            if relabel_nodes:
                _relabelled_graph = self.internal_node_mapping(self._two_player_game)
                _relabelled_graph.plot_graph()
            else:
                self._two_player_game.plot_graph()
            print("Done plotting")

    def build_two_player_implicit_transition_system_from_explicit(self,
                                                                  plot_two_player_implicit_game: bool = False,
                                                                  relabel_nodes: bool = True):
        """
        A helper method to construct an Abstraction in which we do bound the number of times the human can intervene.

        Thus, when a human intervenes you evolve to a Sys state in the same sub-graph. i.e there are no counters on the
        states in the games that indicate the remaining human interventions.
        """

        _graph_name = "two_player_implicit" + self._causal_graph.task.name
        _config_yaml = "/config/" + "two_player_implicit" + self._causal_graph.task.name

        self._two_player_implicit_game = graph_factory.get("TwoPlayerGraph",
                                                           graph_name=_graph_name,
                                                           config_yaml=_config_yaml,
                                                           save_flag=True,
                                                           pre_built=False,
                                                           plot=False)

        # iterate through all the states that have counter i = max_human_intervention. Then Make a copy of that node
        # without the human_intervention counter node[0], look at its neighbour and add them to the graph too similarly.
        # if you are at a human state and human intervenes, then add that state too without the counter.

        for _n in self._two_player_game._graph.nodes():
            # restrict ourself to nodes with a fixed counter
            if _n[1] == self._human_interventions:
                _org_node = _n[0]
                _org_node_attrs = self._two_player_game._graph.nodes[_n]

                if not self._two_player_implicit_game._graph.has_node(_org_node):
                    self._two_player_implicit_game.add_state(_org_node, **_org_node_attrs)

                # look at it successors, add that successor and the corresponding edge
                for _succ in self._two_player_game._graph.successors(_n):
                    _org_succ = _succ[0]
                    _org_succ_attrs = self._two_player_game._graph.nodes[_succ]

                    if not self._two_player_implicit_game._graph.has_node(_org_succ):
                        self._two_player_implicit_game.add_state(_org_succ, **_org_succ_attrs)

                    _edge_attrs = self._two_player_game._graph.edges[_n, _succ, 0]

                    if not self._two_player_implicit_game._graph.has_edge(_org_node, _org_succ):
                        self._two_player_implicit_game.add_edge(u=_org_node,
                                                                v=_org_succ,
                                                                **_edge_attrs)

        # coping so as to avoid dynamic dictionary change errors
        _two_player_implicit_game_copy = copy.deepcopy(self._two_player_implicit_game)

        # human could intervene and evolve to state that oly exists in the sub-graphs after intervening at least once.
        # we iterate through the two_player_implicit_game, see if any states has zero outgoing edge. We then look for
        # its counterpart in the graph with _max_human_counter - 1 state counter, look at its neighbour, add them and
        # their edge
        _human_int = self._human_interventions - 1
        for _n in _two_player_implicit_game_copy._graph.nodes():
            if len(list(_two_player_implicit_game_copy._graph.successors(_n))) == 0:
                for _succ in self._two_player_game._graph.successors((_n, _human_int)):
                    _org_succ = _succ[0]
                    _org_succ_attrs = self._two_player_game._graph.nodes[_succ]

                    if not self._two_player_implicit_game._graph.has_node(_org_succ):
                        self._two_player_implicit_game.add_state(_org_succ, **_org_succ_attrs)

                    _edge_attrs = self._two_player_game._graph.edges[(_n, _human_int), _succ, 0]

                    if not self._two_player_implicit_game._graph.has_edge(_n, _org_succ):
                        self._two_player_implicit_game.add_edge(u=_n,
                                                                v=_org_succ,
                                                                **_edge_attrs)

                    for _succ_of_succ in self._two_player_game._graph.successors(_succ):
                        # if _succ_of_succ[1] == _human_int:
                        _org_succ_of_succ = _succ_of_succ[0]
                        _org_attrs = self._two_player_game._graph.nodes[_succ_of_succ]

                        if not self._two_player_implicit_game._graph.has_node(_org_succ_of_succ):
                            self._two_player_implicit_game.add_state(_org_succ_of_succ, **_org_attrs)
                            warnings.warn("This should not happen")

                        _edge_attrs = self._two_player_game._graph.edges[_succ, _succ_of_succ, 0]

                        if not self._two_player_implicit_game._graph.has_edge(_org_succ, _org_succ_of_succ):
                            self._two_player_implicit_game.add_edge(u=_org_succ,
                                                                    v=_org_succ_of_succ,
                                                                    **_edge_attrs)

        if plot_two_player_implicit_game:
            if relabel_nodes:
                _relabelled_graph = self.internal_node_mapping(self._two_player_implicit_game)
                _relabelled_graph.plot_graph()
            else:
                self._two_player_implicit_game.plot_graph()
            print("Done plotting")

    def _add_valid_human_edges(self,
                               human_state_name: tuple,
                               org_succ_state_name: tuple,
                               human_intervention_cost: int,
                               arch_construction: bool):
        """
        A helper method that adds valid human intervention edges given the current human state, and the original
        successor state if the human decided not to intervene.

        :param human_state_name: The human state which is a tuple. It contains the current configuration of the world
         as an attribute in list and sting format i.e list_ap and ap respectively

        :param org_succ_state_name: The original successor state that game would have evolved if human did not intervene

        :param arch_construction: If this flag, thta mean we are constructing an arch. For the arch building scenario,
        I have fixed the support as well as the top locations. l8 and l9 are support locations and the location on top
        of these is l0 while l3 and l2 are support locations for l1. Thus l2 and l1 are reserved locations for b0 which
        can only go on top. Thus we ignore any human intervention when the arch is built or the robot is about to drop
        a box at the top location.

        This function gets all the valid actions for human intervention given the current robot action,
        world configuration, and evolves the game as per the intervention.
        """
        _human_node: dict = self._two_player_game._graph.nodes[human_state_name]
        _org_succ_node: dict = self._two_player_game._graph.nodes[org_succ_state_name]
        _succ_world_conf: list = _org_succ_node["list_ap"]
        _valid_human_actions: list = self.__get_all_valid_human_intervention(human_node=_human_node,
                                                                             org_succ_node=_org_succ_node,
                                                                             arch_construction=arch_construction)
        _curr_succ_idx: int = org_succ_state_name[1]

        # now add that human edge to the transition system and accordingly update the list_ap and ap attributes of the
        # system node

        for _human_action in _valid_human_actions:
            _box_id, _box_loc = self._get_multiple_box_location(_human_action)

            _succ_node_lbl = _succ_world_conf.copy()
            _succ_node_lbl[_box_id] = _box_loc[1]
            _succ_node_lbl_str = self._convert_list_ap_to_str(_succ_node_lbl)

            _causal_succ_node = _org_succ_node["causal_state_name"]
            _succ_state_name = _causal_succ_node + _succ_node_lbl_str

            _succ_game_state_name = (_succ_state_name, _curr_succ_idx - 1)

            # this action is need to add state/configuration that are only possible because human intervention
            # e.g. human moved a box that the robot was transiting to. The single player ts does not capture such a conf
            # because the raw_pddl_ts does have any transition for robot moving towards an empty location.
            if not self._two_player_game._graph.has_node(_succ_game_state_name):
                self._two_player_game.add_state(_succ_game_state_name,
                                                **_org_succ_node)
                self._two_player_game._graph.nodes[_succ_game_state_name]["list_ap"] = _succ_node_lbl
                self._two_player_game._graph.nodes[_succ_game_state_name]["ap"] = _succ_node_lbl_str

            if not self._two_player_game._graph.has_edge(human_state_name, _succ_game_state_name):
                self._two_player_game.add_edge(u=human_state_name,
                                               v=_succ_game_state_name,
                                               actions=_human_action,
                                               weight=human_intervention_cost)

    def __get_all_valid_human_intervention(self, human_node: dict, org_succ_node: dict, arch_construction: bool ) -> list:
        """
        A helper function that looks up the valid human actions in the causal graph and validate those intervention
        given then current configuration of the world.

        Validity:

        transfer: human has no restriction on how they can move objects around.
        transit: human has no restriction on how they can move objects around except for the one in Robot's hand.
        grasp: human can not move the object currently being picked up/grasped.
        release: human has no restriction on how they can move objects around.
        """

        # if org succ node's causal state name is "holding b#" then the robot is trying to grasp that box.
        _succ_causal_state_name = org_succ_node["causal_state_name"]

        # given a configuration [l0, l1, l2, free] get all the human moves from causal state "on b0 l0" and so on and so
        # forth

        _possible_human_action: list = []
        _current_world_conf: list = human_node["list_ap"]

        # the end effector is currently free
        if _current_world_conf[-1] == "free":
            # the end effector is not performing a grab action
            if "holding" not in _succ_causal_state_name:
                _possible_human_action: list = \
                    self.__get_valid_human_actions_under_transit(current_world_conf=_current_world_conf,
                                                                 arch_construction=arch_construction)
            # the end effector is performing a grab.
            else:
                _possible_human_action: list = \
                    self.__get_valid_human_actions_under_grasp(succ_causal_state_name=_succ_causal_state_name,
                                                               current_world_conf=_current_world_conf)

        # if the robot is holding is an object
        elif "gripper" in _current_world_conf:
            # if the robot is transferring an object
            _transfer_action: bool = False
            for _box in self._causal_graph.task_objects:
                if _box == _current_world_conf[-1]:
                    _transfer_action = True
                    break

            if _transfer_action:
                _possible_human_action: list = \
                    self.__get_valid_human_actions_under_transfer(current_world_conf=_current_world_conf)
            else:
                _possible_human_action: list = \
                    self.__get_valid_human_actions_under_release(current_world_conf=_current_world_conf,
                                                                 arch_construction=arch_construction)

        return _possible_human_action

    def __get_valid_human_actions_under_transit(self, current_world_conf: list, arch_construction: bool) -> list:
        """
        A function that returns a list all possible human action when the robot is trying to perform a transit action
        """
        _valid_human_actions: list = []

        # human cannot intervene once the arch is build or a box is at location l0 or l1
        if arch_construction:
            if "l0" in current_world_conf or "l1" in current_world_conf:
                return _valid_human_actions

        for _box_idx, _box_loc in enumerate(current_world_conf):
            if _box_idx != len(current_world_conf) - 1:
                _state = f"(on b{_box_idx} {_box_loc})"

                # check if this is a valid human action or not by checking if the add_effect
                # (predicate that becomes true)is possible given the current configuration of the world
                if self._causal_graph.causal_graph._graph.has_node(_state):
                    for _succ_node in self._causal_graph.causal_graph._graph[_state]:
                        if _succ_node == _state:
                            continue
                        _add_effect: str = tuple(
                            self._causal_graph.causal_graph._graph[_state][_succ_node][0]["add_effects"])[0]

                        # get the box location where it is being moved to
                        _, _box_loc = self._get_box_location(_add_effect)

                        # if a box is already at this location then this is not a valid human action
                        if _box_loc in current_world_conf:
                            pass
                        else:
                            _valid_human_actions.append(
                                self._causal_graph.causal_graph._graph[_state][_succ_node][0]["actions"])

        return _valid_human_actions

    def __get_valid_human_actions_under_grasp(self, succ_causal_state_name: str, current_world_conf: list) -> list:
        """
        A function that returns a list of all possible human actions when the robot is trying to perform a grasp action
        """

        _valid_human_actions: list = []

        _box_id, _ = self._get_box_location(succ_causal_state_name)

        for _box_idx, _box_loc in enumerate(current_world_conf):
            if _box_idx != len(current_world_conf) - 1 and _box_id != _box_idx:
                _state = f"(on b{_box_idx} {_box_loc})"

                # check if this is a valid human action or not by checking if the add_effect
                # (predicate that becomes)is possible given the current configuration of the world
                for _succ_node in self._causal_graph.causal_graph._graph[_state]:
                    if _succ_node == _state:
                        continue
                    _add_effect: str = tuple(
                        self._causal_graph.causal_graph._graph[_state][_succ_node][0]["add_effects"])[0]

                    # get the box location where it is being moved to
                    _, _box_loc = self._get_box_location(_add_effect)

                    # if a box is already at this location then this is not a valid human action
                    if _box_loc in current_world_conf:
                        pass
                    else:
                        _valid_human_actions.append(
                            self._causal_graph.causal_graph._graph[_state][_succ_node][0]["actions"])

        return _valid_human_actions

    def __get_valid_human_actions_under_transfer(self, current_world_conf: list) -> list:
        """
        A function that returns a list of all possible human actions when the robot is moving a box around
        """
        _valid_human_actions: list = []

        for _box_idx, _box_loc in enumerate(current_world_conf):
            if _box_loc != "gripper" and _box_idx != len(current_world_conf) - 1:
                _state = f"(on b{_box_idx} {_box_loc})"

                # check if this is a valid human action or not by checking if the add_effect
                # (predicate that becomes)is possible given the current configuration of the world
                for _succ_node in self._causal_graph.causal_graph._graph[_state]:
                    if _succ_node == _state:
                        continue
                    _add_effect: str = tuple(
                        self._causal_graph.causal_graph._graph[_state][_succ_node][0]["add_effects"])[0]

                    # get the box location where it is being moved to
                    _, _box_loc = self._get_box_location(_add_effect)

                    # if a box is already at this location then this is not a valid human action
                    if _box_loc in current_world_conf:
                        pass
                    else:
                        _valid_human_actions.append(
                            self._causal_graph.causal_graph._graph[_state][_succ_node][0]["actions"])

        return _valid_human_actions

    def __get_valid_human_actions_under_release(self, current_world_conf: list, arch_construction: bool):
        """
        A function that returns a list of all possible human actions when the robot is trying to drop an object
        """
        _valid_human_actions: list = []

        # human cannot intervene once the arch is build or a box is at location l0 or l1
        if arch_construction:
            if "l0" in current_world_conf or "l1" in current_world_conf:
                return _valid_human_actions

        for _box_idx, _box_loc in enumerate(current_world_conf):
            if _box_loc != "gripper" and _box_idx != len(current_world_conf) - 1:
                _state = f"(on b{_box_idx} {_box_loc})"

                # check if this is a valid human action or not by checking if the add_effect
                # (predicate that becomes)is possible given the current configuration of the world
                for _succ_node in self._causal_graph.causal_graph._graph[_state]:
                    if _succ_node == _state:
                        continue
                    _add_effect: str = tuple(
                        self._causal_graph.causal_graph._graph[_state][_succ_node][0]["add_effects"])[0]

                    # get the box location where it is being moved to
                    _, _box_loc = self._get_box_location(_add_effect)

                    # if a box is already at this location then this is not a valid human action
                    if _box_loc in current_world_conf:
                        pass
                    else:
                        _valid_human_actions.append(
                            self._causal_graph.causal_graph._graph[_state][_succ_node][0]["actions"])

        return _valid_human_actions

    def __add_transition_from_new_sys_states(self, print_new_states: bool = False, arch_construction: bool = False):
        """
        A helper method that identifies states that were created because of human interventions. We then add valid
        Sys transitions from these states.
        """
        for _n in self._two_player_game._graph.nodes():
            if self._two_player_game._graph.out_degree(_n) == 0:
                if print_new_states:
                    print(_n)
                _curr_two_player_node = self._two_player_game._graph.nodes[_n]
                _curr_world_confg = _curr_two_player_node.get("list_ap")
                _curr_world_confg_str = _curr_two_player_node.get("ap")
                _curr_causal_state_name = _curr_two_player_node.get("causal_state_name")
                _curr_box_id, _curr_robo_loc = self._get_box_location(_curr_causal_state_name)
                _intervention_remaining: int = _n[1]
                _cost_dict: Dict = self._transition_system.action_to_cost

                # if its a to-obj action
                if "to-obj" in _n[0]:
                    # from this state we add valid transition to "to-obj b# l#" sys states. These state should
                    # already exists in the two_player_pddl_ts graph
                    _transit_cost: int = _cost_dict.get("transit")
                    for _box_id, _box_loc in enumerate(_curr_world_confg):
                        if _box_id != len(_curr_world_confg) - 1:
                            _valid_state_to_transit: tuple = (f'(to-obj b{_box_id} {_box_loc}){_curr_world_confg_str}',
                                                              _intervention_remaining)

                            if not self._two_player_game._graph.has_node(_valid_state_to_transit):
                                warnings.warn(f"Adding a transition from {_n} to {_valid_state_to_transit}."
                                              f" The state {_valid_state_to_transit} does not already exist")

                            _edge_action = f"transit b{_box_id} {_curr_robo_loc} {_box_loc}"
                            self._two_player_game.add_edge(u=_n,
                                                           v=_valid_state_to_transit,
                                                           actions=_edge_action,
                                                           weight=_transit_cost)
                elif "to-loc" in _n[0]:
                    # in this state the robot is moving a box. So, we add transitions to location that are currently
                    # available/free
                    _transfer_cost: int = _cost_dict.get("transfer")
                    _occupied_locs: set = set()
                    _succ_world_conf = _curr_world_confg.copy()
                    for _idx, _loc in enumerate(_curr_world_confg):
                        if _idx != len(_curr_world_confg) - 1 and _loc != "gripper":
                            _occupied_locs.add(_loc)

                    _free_loc = set(self._causal_graph.task_locations) - _occupied_locs

                    for _loc in _free_loc:
                        _succ_world_conf[-1] = _loc
                        _succ_world_conf_str = self._convert_list_ap_to_str(ap=_succ_world_conf)
                        _valid_state_to_transit: tuple = (f'(to-loc b{_curr_box_id} {_loc}){_succ_world_conf_str}',
                                                          _intervention_remaining)

                        if not self._two_player_game._graph.has_node(_valid_state_to_transit):
                            warnings.warn(f"Adding a transition from {_n} to {_valid_state_to_transit}."
                                          f" The state {_valid_state_to_transit} does not already exist")

                        _edge_action = f"transfer b{_curr_box_id} {_curr_robo_loc} {_loc}"
                        self._two_player_game.add_edge(u=_n,
                                                       v=_valid_state_to_transit,
                                                       actions=_edge_action,
                                                       weight=_transfer_cost)

                else:
                    if not arch_construction:
                        warnings.warn(f"Encountered a Sys state due to human intervention which was unaccounted for. "
                                      f" The Sys state is {_n}")
                    else:
                        if "holding" in _n[0]:
                            # from holding state you can transfer to another "to-loc b0 {empty loc states}"
                            _transfer_cost: int = _cost_dict.get("transfer")
                            _occupied_locs: set = set()
                            _succ_world_conf = _curr_world_confg.copy()
                            for _idx, _loc in enumerate(_curr_world_confg):
                                if _idx != len(_curr_world_confg) - 1 and _loc != "gripper":
                                    _occupied_locs.add(_loc)

                            _free_loc = set(self._causal_graph.task_locations) - _occupied_locs

                            for _loc in _free_loc:
                                _succ_world_conf[-1] = _loc
                                _succ_world_conf_str = self._convert_list_ap_to_str(ap=_succ_world_conf)
                                # this could be to-loc b0 {_loc}. But that's fine
                                _valid_state_to_transit: tuple = (
                                f'(to-loc b{_curr_box_id} {_loc}){_succ_world_conf_str}',
                                _intervention_remaining)

                                if not self._two_player_game._graph.has_node(_valid_state_to_transit):
                                    warnings.warn(f"Adding a transition from {_n} to {_valid_state_to_transit}."
                                                  f" The state {_valid_state_to_transit} does not already exist")

                                _edge_action = f"transfer b{_curr_box_id} {_curr_robo_loc} {_loc}"
                                self._two_player_game.add_edge(u=_n,
                                                               v=_valid_state_to_transit,
                                                               actions=_edge_action,
                                                               weight=_transfer_cost)
                        else:
                            warnings.warn(
                                f"Encountered a Sys state due to human intervention which was unaccounted for during"
                                f" arch construction abstraction. The Sys state is {_n}")


    def _get_multiple_box_location(self, multiple_box_location_str: str) -> Tuple[int, List[str]]:
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

    def _convert_list_ap_to_str(self, ap: list, separator='_') -> str:
        """
        A helper method to convert a state label/atomic proposition which is in a list of elements into a str

        :param ap: Atomic proposition of type list
        :param separator: element used to join the elements in the given list @ap

        ap: ['l3', 'l4', 'l1', 'free']
        _ap_str = 'l3_l4_l1_free'
        """
        if not isinstance(ap, list):
            warnings.warn(f"Trying to convert an atomic proposition of type {type(ap)} into a string.")

        _ap_str = separator.join(ap)

        return _ap_str

    def _get_box_location(self, box_location_state_str: str) -> Tuple[int, str]:
        """
        A function that returns the location of the box and the box id in the given world.

        e.g Str: on b# l1 then return l1

        NOTE: The string should be exactly in the above formation i.e on<whitespace>b#<whitespave>l#. We can swap
         between small and capital i.e 'l' & 'L' are valid.
        """

        _loc_pattern = "[l|L][\d]+"
        try:
            _loc_state: str = re.search(_loc_pattern, box_location_state_str).group()
        except AttributeError:
            _loc_state = ""
            print(f"The causal_state_string {box_location_state_str} dose not contain location of the box")

        _box_pattern = "[b|B][\d]+"
        try:
            _box_state: str = re.search(_box_pattern, box_location_state_str).group()
        except AttributeError:
            _box_state = ""
            print(f"The causal_state_string {box_location_state_str} dose not contain box id")

        _box_id_pattern = "\d+"
        _box_id: int = int(re.search(_box_id_pattern, _box_state).group())

        return _box_id, _loc_state

    def internal_node_mapping(self, game: TwoPlayerGraph) -> TwoPlayerGraph:
        """
        A helper function that created a node to int dictionary. This helps in plotting as the node names in
        two_player_pddl_ts_game are huge.
        """

        _node_int_map = bidict({state: index for index, state in enumerate(game._graph.nodes)})
        _modified_two_player_pddl_ts = copy.deepcopy(game)

        _relabelled_graph = nx.relabel_nodes(game._graph, _node_int_map, copy=True)
        _modified_two_player_pddl_ts._graph = _relabelled_graph

        return _modified_two_player_pddl_ts

    def set_appropriate_ap_attribute_name(self, implicit: bool = True):
        """
        A helper function that iterates through every node in the two player game, removes the list ap attribute
        and replaces the ap attribute with that list ap. We also add a new node attribute str_ap that stores the string
        form of the list ap attribute corresponding to that node.
        """

        if implicit:
            game = self._two_player_implicit_game
        else:
            game = self._two_player_game

        for _n in game._graph.nodes():
            _node_atts = game._graph.nodes[_n]
            _tmp_ap = _node_atts.get("list_ap")
            _tmp_str_ap = _node_atts.get("ap")

            game._graph.nodes[_n]['ap'] = _tmp_ap
            game._graph.nodes[_n]['str_ap'] = _tmp_str_ap

            # delete the list_ap node attribute
            del game._graph.nodes[_n]['list_ap']

    def modify_ap_w_object_types(self, implicit: bool = True):
        """
        A function that modifies the list of atomic propositions that are true at a given state with the box type

        e.g ["l2", "l3", "l4", "free"] => ["p02", "p13", "p24", "free"] or
        [gripper, "l3", "l0", "b0"] => ["gripper", "p13", "p20", "b0"].

        NOTE: Before calling this function, make sure we call the set_appropriate_ap_attribute_name() method that swaps
         the list_ap node with ap attribute.
        """
        if implicit:
            game = self._two_player_implicit_game
        else:
            game = self._two_player_game

        for _n in game._graph.nodes():
            _list_ap = game.get_state_w_attribute(_n, "ap")
            _tmp_lst_ap = _list_ap.copy()

            for _idx, _box_loc in enumerate(_list_ap):
                if _box_loc == "gripper" or _idx == len(_list_ap) - 1:
                    continue
                else:
                    _loc = re.findall('[0-9]+', _box_loc)
                    _new_ap_str = f"p{_idx}{_loc[0]}"
                    _tmp_lst_ap[_idx] = _new_ap_str

            game._graph.nodes[_n]['ap'] = _tmp_lst_ap

    def build_LTL_automaton(self, formula: str, debug: bool = False, plot: bool = False, use_alias: bool = False):
        """
        A method to construct automata using the regret_synthesis_tool.
        """
        self._formula = formula

        if not isinstance(formula, str):
            warnings.warn("Please make sure the input formula is of type string.")

        _ltl_automaton = graph_factory.get('DFA',
                                           graph_name="pddl_ltl",
                                           config_yaml="/config/pddl_ltl",
                                           save_flag=True,
                                           sc_ltl=formula,
                                           use_alias=use_alias,
                                           plot=plot)

        if debug:
            print(f"The pddl formula is : {formula}")

        return _ltl_automaton

    def build_product(self, dfa, trans_sys, plot: bool = False):
        _product_automaton = graph_factory.get("ProductGraph",
                                               graph_name="pddl_product_graph",
                                               config_yaml="/config/pddl_product_graph",
                                               trans_sys=trans_sys,
                                               automaton=dfa,
                                               # dfa=dfa,
                                               save_flag=True,
                                               prune=False,
                                               debug=False,
                                               absorbing=True,
                                               finite=False,
                                               plot=plot)

        print("Done building the Product Automaton")

        # Add the accepting state "accept_all" in the product graph with player = "eve"
        # should technically be only one if absorbing is true
        _states = _product_automaton.get_accepting_states()

        for _s in _states:
            _product_automaton.add_state_attribute(_s,
                                                   attribute_key="player",
                                                   attribute_value="eve")

        return _product_automaton

