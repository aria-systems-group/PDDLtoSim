import re
import copy
import warnings

import networkx as nx

from typing import Tuple, Dict, List, Optional
from collections import deque, defaultdict
from bidict import bidict

from regret_synthesis_toolbox.src.graph import graph_factory
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph, ProductAutomaton

# import local packages
from .causal_graph import CausalGraph
from .transition_system import FiniteTransitionSystem


class TwoPlayerGame:
    """
    A Class that builds a Two player game based on the Transition System built using the Causal Graph.
    """

    def __init__(self, causal_graph, transition_system, arch_locs_dict: dict = None):
        self._causal_graph: CausalGraph = causal_graph
        self._transition_system: FiniteTransitionSystem = transition_system
        self._two_player_game: Optional[TwoPlayerGraph] = None
        self._two_player_implicit_game: Optional[TwoPlayerGraph] = None
        self._arch_locs_dict = arch_locs_dict

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
    

    def construct_game_from_yaml(self, graph_yaml_dir: dict):
        """
         A helper function to load the game from yaml file
        """
        _graph_name = "two_player_implicit" + self._causal_graph.task.name
        _config_yaml = "/config/" + "two_player_implicit_" + self._causal_graph.task.name

        self._two_player_implicit_game = graph_factory.get("TwoPlayerGraph",
                                                           graph_name=_graph_name,
                                                           config_yaml=graph_yaml_dir,
                                                           from_file=True,
                                                           save_flag=True,
                                                           plot=False)

    def construct_env_node_for_new_sys_player_states(self,
                                                    arch_construction: bool = False,
                                                    human_intervention_cost: int = 0,
                                                    human_non_intervention_cost: int = 0,
                                                    new_states_created: set = set({})):
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
            
            assert self._two_player_game.get_state_w_attribute(_u, "player") == 'eve', \
            "[Error] Error while constructing the 2-player bounded game. Addidng new edges from Env player. "
            "This should not happen. Fix the code" 
            assert _u in new_states_created or _v in new_states_created, "[Error] Error while constructing the 2-player bounded game. Came across as state unaccounted for. Fix This!! " 
            
            _edge_action = self._two_player_game._graph.get_edge_data(*_e)[0]['actions']

            _env_node = (f"h{_u[0]}{_edge_action}", i)
            # adam_node_lst.append(_env_node)

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

        new_states_created = self.__add_transition_from_new_sys_states(print_new_states=False, arch_construction=arch_construction)
        count = 0
        while len(new_states_created) > 0:
            if count == 0:
                print("Iterating to check if human interventions created any new nodes")
            print(f"Iteration Count: {count + 1}")
            self.construct_env_node_for_new_sys_player_states(arch_construction=arch_construction,
                                                              human_intervention_cost=human_intervention_cost,
                                                              human_non_intervention_cost=human_non_intervention_cost,
                                                              new_states_created=new_states_created)
            new_states_created = self.__add_transition_from_new_sys_states(print_new_states=False, arch_construction=arch_construction)
            print(f"Debugging: {len(new_states_created)} - # of new states")
            count += 1

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
        _config_yaml = "/config/" + "two_player_implicit_" + self._causal_graph.task.name
        _init_state = self._transition_system.transition_system.get_initial_states()[0][0]

        self._two_player_implicit_game = graph_factory.get("TwoPlayerGraph",
                                                           graph_name=_graph_name,
                                                           config_yaml=_config_yaml,
                                                           save_flag=True,
                                                           plot=False)

        # iterate through all the states that have counter i = max_human_intervention. Then Make a copy of that node
        # without the human_intervention counter node[0], look at its neighbour and add them to the graph too similarly.
        # if you are at a human state and human intervenes, then add that state too without the counter.
        # num_warnings = 0
        for _n in self._two_player_game._graph.nodes():
            # restrict ourself to nodes with a fixed counter
            # if _n[1] == self._human_interventions:
            _org_node = _n[0]
            _org_node_attrs = self._two_player_game._graph.nodes[_n]

            if not self._two_player_implicit_game._graph.has_node(_org_node):
                self._two_player_implicit_game.add_state(_org_node, **_org_node_attrs)
            
            # adding it form attr dictionary does not work because
            # you n copies of the init state and only the nth copy is the init state. 
            if _org_node == _init_state and _n[1] == self._human_interventions:
                self._two_player_implicit_game._graph.nodes[_org_node]['init'] = True

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
        # _two_player_implicit_game_copy = copy.deepcopy(self._two_player_implicit_game)

        # human could intervene and evolve to state that only exists in the sub-graphs after intervening at least once.
        # we iterate through the two_player_implicit_game, see if any states has zero outgoing edge. We then look for
        # its counterpart in the graph with _max_human_counter - 1 state counter, look at its neighbour, add them and
        # their edge
        # _human_int = self._human_interventions - 1
        # for _n in _two_player_implicit_game_copy._graph.nodes():
        #     if len(list(_two_player_implicit_game_copy._graph.successors(_n))) == 0:
        #         for _succ in self._two_player_game._graph.successors((_n, _human_int)):
        #             _org_succ = _succ[0]
        #             _org_succ_attrs = self._two_player_game._graph.nodes[_succ]

        #             if not self._two_player_implicit_game._graph.has_node(_org_succ):
        #                 self._two_player_implicit_game.add_state(_org_succ, **_org_succ_attrs)

        #             _edge_attrs = self._two_player_game._graph.edges[(_n, _human_int), _succ, 0]

        #             if not self._two_player_implicit_game._graph.has_edge(_n, _org_succ):
        #                 self._two_player_implicit_game.add_edge(u=_n,
        #                                                         v=_org_succ,
        #                                                         **_edge_attrs)

        #             for _succ_of_succ in self._two_player_game._graph.successors(_succ):
        #                 # if _succ_of_succ[1] == _human_int:
        #                 _org_succ_of_succ = _succ_of_succ[0]
        #                 _org_attrs = self._two_player_game._graph.nodes[_succ_of_succ]

        #                 if not self._two_player_implicit_game._graph.has_node(_org_succ_of_succ):
        #                     self._two_player_implicit_game.add_state(_org_succ_of_succ, **_org_attrs)
        #                     warnings.warn("This should not happen")
        #                     num_warnings+= 1

        #                 _edge_attrs = self._two_player_game._graph.edges[_succ, _succ_of_succ, 0]

        #                 if not self._two_player_implicit_game._graph.has_edge(_org_succ, _org_succ_of_succ):
        #                     self._two_player_implicit_game.add_edge(u=_org_succ,
        #                                                             v=_org_succ_of_succ,
        #                                                             **_edge_attrs)
        # print(f"The number of warnings I got are: {num_warnings}")
        num_of_nodes_w_no_outgoing_edges = 0
        for game_node in self._two_player_implicit_game._graph.nodes():
            # Only process nodes with no outgoing edges
            if self._two_player_implicit_game._graph.out_degree(game_node) == 0:
                print(f"State {game_node} has no outgoing edges.")
                num_of_nodes_w_no_outgoing_edges += 1
        print("The number of nodes with no outgoing edges is: ", num_of_nodes_w_no_outgoing_edges)
            

        if plot_two_player_implicit_game:
            if relabel_nodes:
                _relabelled_graph = self.internal_node_mapping(self._two_player_implicit_game)
                _relabelled_graph.plot_graph()
            else:
                self._two_player_implicit_game.plot_graph()
            print("Done plotting")

    def _add_valid_human_edges(self,
                               human_state_name: Tuple[str, int],
                               org_succ_state_name: Tuple[str, int],
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

        if arch_construction and len(_valid_human_actions) > 0:
            _valid_human_actions = self.__get_valid_human_actions_for_arch_constrcution(current_world_config=_human_node['list_ap'],
                                                                                        current_state_name=human_state_name,
                                                                                        succ_state_name=org_succ_state_name,
                                                                                        valid_human_actions=_valid_human_actions)

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
    
    def __get_valid_human_actions_for_arch_constrcution(self, current_world_config, current_state_name, succ_state_name, valid_human_actions: list) -> list:
        """
         A function that returns a list of all possible human actions when the robot is trying to build an arch.
           We call this method the arch_construction flag is true
        """
        supports_not_constructed_dict: Dict[int, bool] = {}

        # if there does not exists  boxes in support location then the human can not move a box on top.
        for num, arch_locs in self._arch_locs_dict.items():
            supports_not_constructed_dict[num] = False
            for loc in arch_locs['supports']:
                if loc not in current_world_config:
                    supports_not_constructed_dict[num] = True
                    break
        # there are conf where the robot is about to release the object in the arch support locs.
        #  While the above loop returns true the arch has not been completed yet.
        # for num, arch_locs in self._arch_locs_dict.items():
        #     if not supports_not_constructed_dict[num] and 'free' not in current_world_config:
        #         supports_not_constructed_dict[num] = True
            
        actions_to_remove = set()
        for num, supports_not_constructed in supports_not_constructed_dict.items():
            if supports_not_constructed:
                for human_move in valid_human_actions:
                    _, box_loc = self._get_multiple_box_location(human_move)
                    
                    # if support not constructed then human can not move box on top
                    if box_loc[1] in self._arch_locs_dict[num]['top']:
                        actions_to_remove.add(human_move)
            else:
                for human_move in valid_human_actions:
                    _, box_loc = self._get_multiple_box_location(human_move)
                    # if not supports_not_constructed (aka supports constructed) but the top is not occupied, then human can only move the a non-support block to top
                    if self._arch_locs_dict[num]['top'] not in current_world_config:
                        if box_loc[0] in self._arch_locs_dict[num]['supports'] and box_loc[1] == self._arch_locs_dict[num]['top']:
                            actions_to_remove.add(human_move)

                    # if the support are constructed and top locs is also occupied (aka the arch is constrcuted) then human can remove top loc
                    elif self._arch_locs_dict[num]['top'] in current_world_config and box_loc[0] != self._arch_locs_dict[num]['top']:
                        actions_to_remove.add(human_move)
        
        valid_human_actions = [action for action in valid_human_actions if action not in actions_to_remove]

        # during arch constuction we do have edges of the form ('(holding b2 l4)l6_l5_gripper_b2', 1) -> (("h('(holding b2 l4)l6_l5_gripper_b2', 1)release b2 l4", 1))) -> ('(ready l4)l6_l5_l4_free', 1)
        if 'holding' in current_state_name[0].split(')', 1)[0] and 'ready' in succ_state_name[0].split(')', 1)[0]: 
            _, robo_loc = self._get_box_location(succ_state_name[0].split(')', 1)[0] + ")")  # will return l4 form (ready l4)
            
            for num, supports_not_constructed in supports_not_constructed_dict.items():
                assert supports_not_constructed, "[Error] Removing human edges while the arch has NOT been formed in the next Sys state. Fix this!!!"
                assert self._arch_locs_dict[num]['top'] in current_world_config, "[Error] Removing human edges while the arch has NOT been formed in the next Sys state. Fix this!!!"
                
                # this is not captured by the traditional compute human intervention method as the the game is directly evolving from holding state to ready state and skipping to-loc
                for human_move in valid_human_actions:
                    _, box_loc = self._get_multiple_box_location(human_move)
                    # remove human actions that are moving a box to loc where the robot is trying to move.
                    if box_loc[1] == robo_loc:
                        actions_to_remove.add(human_move)
                    # remove human actions that are moving a support box while the top loc is occupied.
                    elif box_loc[0] in self._arch_locs_dict[num]['supports']: 
                        actions_to_remove.add(human_move)
        
        valid_human_actions = [action for action in valid_human_actions if action not in actions_to_remove]

        return valid_human_actions


    def __get_valid_human_actions_under_transit(self, current_world_conf: list, arch_construction: bool) -> list:
        """
        A function that returns a list all possible human action when the robot is trying to perform a transit action
        """
        _valid_human_actions: list = []

        # human cannot intervene once the arch is build or a box is at location l0 or l1
        # if arch_construction:
        #     # if "l0" in current_world_conf or "l1" in current_world_conf:
        #     for arch_locs in self._arch_locs_dict.values():
        #         if arch_locs['top'] in current_world_conf:
        #             return _valid_human_actions

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
        # if arch_construction:
        #     # if "l0" in current_world_conf or "l1" in current_world_conf:
        #     #     return _valid_human_actions
        #     for arch_locs in self._arch_locs_dict.values():
        #         if arch_locs['top'] in current_world_conf:
        #             return _valid_human_actions

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


    def _get_node_info(self, node):
        """
        Extract and return common node information.
        """
        node_info = {
            'curr_two_player_node': self._two_player_game._graph.nodes[node],
            'curr_world_config': self._two_player_game._graph.nodes[node].get("list_ap"),
            'curr_world_config_str': self._two_player_game._graph.nodes[node].get("ap"),
            'curr_causal_state_name': self._two_player_game._graph.nodes[node].get("causal_state_name"),
            'intervention_remaining': node[1]
        }
        
        # Extract box ID and robot location
        # makin it more robust 
        
        box_id, robo_loc = self._get_box_location(node_info['curr_causal_state_name'])
        if box_id == "":
            assert "ready" in node_info['curr_causal_state_name'], "[Error] Error when  extracting box id and location for ready state. FIX THIS!!!!"
            box_id = node_info['curr_world_config'].index(robo_loc)

        node_info['curr_box_id'] = box_id
        node_info['curr_robo_loc'] = robo_loc
        
        return node_info
    

    def _add_edge_with_validation(self, from_node, to_node, action: str, weight: int, new_nodes_list: set) -> set:
        """
        Add an edge with validation to ensure the target node exists.
        """
        if not self._two_player_game._graph.has_node(to_node):
            # warnings.warn(f"Adding a transition from {from_node} to {to_node}. "
                        # f"The state {to_node} does not already exist")
            
            # Create the node with attributes inherited from the source node's successor
            causal_state_name = to_node[0].split(')')[0] + ')'
            world_config_str = to_node[0].split(')')[1]
            target_attrs = {
                'causal_state_name': causal_state_name,
                'list_ap': world_config_str.split('_'),
                'ap': world_config_str, 
                'player': 'eve'}
            self._two_player_game.add_state(to_node, **target_attrs)
            new_nodes_list.add(to_node)
        
        # Add the edge
        if not self._two_player_game._graph.has_edge(from_node, to_node):
            self._two_player_game.add_edge(u=from_node,
                                           v=to_node,
                                           actions=action,
                                           weight=weight)
    

    def _add_sys_transitions_for_to_obj_node(self, node, node_info: dict, new_nodes_list: set) -> set:
        """
        Add transitions from a 'to-obj' Sys node to valid Sys successor states.

        @param node: The current node in the graph
        @param node_info: A dictionary containing information about the current node
        @param new_nodes_list: A set to keep track of newly created nodes
        @return: None
        """
        transit_cost: int = self._transition_system.action_to_cost.get("transit")
        curr_robo_loc = node_info['curr_robo_loc']
        curr_world_config = node_info['curr_world_config']
        curr_world_config_str = node_info['curr_world_config_str']
        intervention_remaining = node_info['intervention_remaining']
        
        # Add valid transitions to "to-obj b# l#" sys states
        for box_id, box_loc in enumerate(curr_world_config[:-1]):
            valid_state = (f'(to-obj b{box_id} {box_loc}){curr_world_config_str}', intervention_remaining)
            edge_action = f"(transit b{box_id} {curr_robo_loc} {box_loc})"
            self._add_edge_with_validation(node, valid_state, edge_action, transit_cost, new_nodes_list)
    

    def _add_sys_transitions_for_to_loc_node(self, node, node_info: dict, new_nodes_list: set) -> set:
        """
        Add transitions from a 'to-loc' Sys node to valid  Sys successor states.

        @param node: The current node in the graph
        @param node_info: A dictionary containing information about the current node
        @param new_nodes_list: A set to keep track of newly created nodes
        @return: None
        """
        transfer_cost: int = self._transition_system.action_to_cost.get("transfer")
        curr_world_config = node_info['curr_world_config']
        curr_box_id = node_info['curr_box_id']
        curr_robo_loc = node_info['curr_robo_loc']
        intervention_remaining = node_info['intervention_remaining']
        
        # Find occupied locations
        succ_world_conf = curr_world_config.copy()
        occupied_locs = {loc for loc in curr_world_config[:-1] if loc != "gripper"}
        
        # Find free locations
        free_locs = set(self._causal_graph.task_locations) - occupied_locs
        
        # Add transitions to free locations
        for loc in free_locs:
            succ_world_conf[-1] = loc
            succ_world_conf_str = self._convert_list_ap_to_str(ap=succ_world_conf)
            valid_state = (f'(to-loc b{curr_box_id} {loc}){succ_world_conf_str}', intervention_remaining)
            edge_action = f"(transfer b{curr_box_id} {curr_robo_loc} {loc})"
            self._add_edge_with_validation(node, valid_state, edge_action, transfer_cost, new_nodes_list)
    

    def _add_arch_release_transitions(self, node, node_info: dict, new_nodes_list: set) -> bool:
        """
         Add transitions for releasing a block when building an arch.
        
        @param node: The current node in the graph
        @param node_info: A dictionary containing information about the current node
        @param new_nodes_list: A set to keep track of newly created nodes
        @return: bool - True if edges were added, False otherwise
        """
        curr_box_id = node_info['curr_box_id']
        curr_robo_loc = node_info['curr_robo_loc']
        curr_world_config = node_info['curr_world_config']
        intervention_remaining = node_info['intervention_remaining']
        release_cost: int = self._transition_system.action_to_cost.get("release")
        added_edges: bool = False
        
        for arch in self._arch_locs_dict.values():
            if curr_robo_loc in arch['supports'] and arch['top'] in curr_world_config:
                # Add a release node at the same place
                succ_world_conf = curr_world_config.copy()
                succ_world_conf[-1] = 'free'
                succ_world_conf[curr_box_id] = curr_robo_loc
                succ_world_conf_str = self._convert_list_ap_to_str(ap=succ_world_conf)
                valid_state = (f'(ready {curr_robo_loc}){succ_world_conf_str}', intervention_remaining)
                self._add_edge_with_validation(node, valid_state, f"(release b{curr_box_id} {curr_robo_loc})", release_cost, new_nodes_list)
                added_edges = True
        
        return added_edges


    def _add_sys_transitions_for_holding_node(self, node, node_info: dict, arch_construction: bool, new_nodes_list: set) -> None:
        """
        Add transitions from a 'holding' node to valid successor states.

        @param node: The current node in the graph
        @param node_info: A dictionary containing information about the current node
        @param new_nodes_list: A set to keep track of newly created nodes
        @return: None
        """
        # First, process any arch-specific release actions
        added_arch_edges = False
        if arch_construction:
            added_arch_edges = self._add_arch_release_transitions(node, node_info, new_nodes_list)
        
        # Then add general transfer actions to empty locations
        if not added_arch_edges:
            self._add_sys_transitions_for_to_loc_node(node, node_info, new_nodes_list)


    def __add_transition_from_new_sys_states(self, print_new_states: bool = False, arch_construction: bool = False) -> set:
        """
        A helper method that identifies states that were created because of human interventions. We then add valid
        Sys transitions from these states.
        """
        _old_two_player_pddl_ts: TwoPlayerGraph = copy.deepcopy(self._two_player_game)
        new_states_created = set()
        for game_node in _old_two_player_pddl_ts._graph.nodes():
            # Only process nodes with no outgoing edges
            if self._two_player_game._graph.out_degree(game_node) == 0:
                new_states_created.add(game_node)
                if print_new_states:
                    print(game_node)
                
                # Extract common node information
                node_info = self._get_node_info(game_node)

                # if its a to-obj action
                if "to-obj" in game_node[0]:
                    # from this state we add valid transition to "to-obj b# l#" sys states. These state should
                    # already exists in the two_player_pddl_ts graph
                    self._add_sys_transitions_for_to_obj_node(game_node, node_info, new_states_created)
                elif "to-loc" in game_node[0]:
                    # in this state the robot is moving a box. So, we add transitions to location that are currently
                    # available/free
                    self._add_sys_transitions_for_to_loc_node(game_node, node_info, new_states_created)

                else:
                    if not arch_construction:
                        warnings.warn(f"Encountered a Sys state due to human intervention which was unaccounted for. "
                                      f" The Sys state is {game_node}")
                    else:
                        if "holding" in game_node[0]:
                            self._add_sys_transitions_for_holding_node(game_node, node_info, arch_construction, new_states_created)
                        elif "ready" in game_node[0]:
                            self._add_sys_transitions_for_to_obj_node(game_node, node_info, new_states_created)
                        else:
                            warnings.warn(f"Encountered a Sys state due to human intervention which was unaccounted for during"
                                          f" arch construction abstraction. The Sys state is {game_node}")
        
        return new_states_created



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
            _box_id_pattern = "\d+"
            _box_id: int = int(re.search(_box_id_pattern, _box_state).group())
        except AttributeError:
            # if the state is ready then the robot is not holding any box
            if 'ready' in box_location_state_str:
                _box_id = ""
            else:
                warnings.warn(f"The causal_state_string {box_location_state_str} dose not contain box id")
        return _box_id, _loc_state

    def internal_node_mapping(self, game: TwoPlayerGraph) -> TwoPlayerGraph:
        """
        A helper function that creates a node to int dictionary. This helps in plotting as the node names in
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

    def modify_edge_weights(self, implicit: bool = True):
        """
        A helper function in which I modify weights corresponding to actions that transit to a safe state from which
        the human cannot intervene. The actions could be evolving from outside to this set or actions that are evolving
        within this set.
        """

        # get the set of locations that are of type - "box-loc"
        _non_intervening_locs = self._causal_graph.task_non_intervening_locations
        _intervening_locs = self._causal_graph.task_intervening_locations

        if implicit:
            game = self._two_player_implicit_game
        else:
            game = self._two_player_game

        # iterate through all edge and multiply the weight by 4 for edges as per the doc string
        for _e in game._graph.edges():
            _u = _e[0]
            _v = _e[1]

            _edge_action = game._graph[_u][_v][0].get('actions')

            if game._graph.nodes[_u]['player'] == 'adam':
                continue

            # get the from and to loc
            _, _locs = self._get_multiple_box_location(_edge_action)
            _from_loc = ""
            _to_loc = ""
            if len(_locs) == 2:
                _from_loc = _locs[0]
                _to_loc = _locs[1]
            else:
                _to_loc = _locs[0]

            # all action within the non_intervening loc are twice as expensive as the other region
            if _to_loc != "" and _from_loc != "":
                if _to_loc in _non_intervening_locs:
                    game._graph[_u][_v][0]['weight'] = 3
                    continue

            if _from_loc == "" and _to_loc in _non_intervening_locs:
                game._graph[_u][_v][0]['weight'] = 3
                continue

    def build_LTL_automaton(self, formula: str, debug: bool = False, plot: bool = False, use_alias: bool = False):
        """
        A method to construct LTL automata using the regret_synthesis_tool.
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
    
    def build_LTLf_automaton(self, formula: str, debug: bool = False, plot: bool = False, use_alias: bool = False):
        """
         A method to construct LTLf automata using the regret_synthesis_tool.
        """
        self._formula = formula

        if not isinstance(formula, str):
            warnings.warn("Please make sure the input formula is of type string.")

        _ltl_automaton = graph_factory.get('LTLfDFA',
                                           graph_name="pddl_ltlf_adm",
                                           config_yaml="/config/pddl_ltlf_adm",
                                           save_flag=True,
                                           ltlf=formula,
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

        # for _s in _states:
        #     _product_automaton.add_state_attribute(_s,
        #                                            attribute_key="player",
        #                                            attribute_value="eve")

        return _product_automaton

