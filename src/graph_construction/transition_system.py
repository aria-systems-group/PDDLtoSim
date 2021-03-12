import sys
import warnings
import re
import os
import copy
import networkx as nx

from collections import deque, defaultdict
from typing import Tuple, Dict, List, Optional

from regret_synthesis_toolbox.src.graph import graph_factory
from regret_synthesis_toolbox.src.graph import FiniteTransSys

# import local packages
from .causal_graph import CausalGraph


class FiniteTransitionSystem:
    """
    A class that build a builds a transition system given a Causal Graph as Input. The Causal Graph is constructed using
    domain and problem file.
    """

    def __init__(self, causal_graph):
        self._causal_graph: CausalGraph = causal_graph
        self._transition_system: Optional[FiniteTransSys] = None
        self._action_to_cost: Optional[Dict] = self._set_default_action_cost_mapping()

    @property
    def transition_system(self):
        if isinstance(self._transition_system, type(None)):
            warnings.warn("The transition system is of type of None. Please build the transition system before"
                          " accessing it")
        return self._transition_system

    @property
    def action_to_cost(self):
        return self._transition_system

    @action_to_cost.setter
    def action_to_cost(self, action_cost_dict):

        for _action, _cost in action_cost_dict:
            if _cost < 0:
                warnings.warn(f"The cost of taking {_action} is {_cost}. The current regret minimizing algorithm only"
                              f" works with >= 0 weights.")

        self._action_to_cost = action_cost_dict

    def _set_default_action_cost_mapping(self) -> Dict:
        """
        The function return the default action to cost mapping. All the action have weight 1
        """

        _action_cost_mapping: Dict[str, int] = \
            {"transit": 1,
             "transfer": 1,
             "grasp": 1,
             "release": 1
             }
        return _action_cost_mapping

    def build_transition_system(self, plot: bool = False):
        """
        A function that builds the transition system given a causal graph.

        Given a Causal graph, we iterate through the states of the graph, starting from the initial state, and updating
        the label (atomic proposition) for each state till we label all the states in the graph.

        The nodes of Transitions System are named:

            1) The current causal state name (indicates the current action being formed) +
            2) The current configuration of the world

        Each node in the Transition System has the following node attributes:
            1) causal_state_name: To keep track of the action the robot is performing
            2) player: To which player does this belong to. This attribute comes handy when constructing the two player
            game based on a Transition System. All nodes in Transition System should belong to "Eve/Sys/Min" player.
            3) list_ap: The current configuration of th world in the form of a list
            4) ap: The current configuration of the world in the form of a str

        """

        _init_state_label, _init_robo_conf = self._get_initial_state_label()

        # lets have two stack - visitedStack and doneStack
        # As you encounter nodes, keep adding them to the visitedStack. As you encounter a neighbour that you already
        # visited, pop that node and add that node to the done stack. Repeat the process till the visitedStack is empty.

        visited_stack = deque()
        done_stack = deque()

        _graph_name = "pddl_ts_" + self._causal_graph.task.name
        _config_yaml = "/config/" + "pddl_ts_" + self._causal_graph.task.name

        self._transition_system: FiniteTransSys = graph_factory.get('TS',
                                                                    raw_trans_sys=None,
                                                                    config_yaml=_config_yaml,
                                                                    graph_name=_graph_name,
                                                                    from_file=False,
                                                                    pre_built=False,
                                                                    save_flag=True,
                                                                    debug=False,
                                                                    plot=False,
                                                                    human_intervention=0,
                                                                    plot_raw_ts=False)

        _causal_current_node = _init_robo_conf
        _str_curr_lbl = self._convert_list_ap_to_str(_init_state_label)

        _game_current_node = _causal_current_node + _str_curr_lbl
        visited_stack.append(_game_current_node)

        self._transition_system.add_state(_game_current_node,
                                          causal_state_name=_causal_current_node,
                                          player="eve",
                                          list_ap=_init_state_label.copy(),
                                          ap=_str_curr_lbl)

        self._transition_system.add_initial_state(_game_current_node)

        while visited_stack:
            _game_current_node = visited_stack.popleft()
            _causal_current_node = self._transition_system._graph.nodes[_game_current_node].get('causal_state_name')

            for _causal_succ_node in self._causal_graph.causal_graph._graph[_causal_current_node]:
                # add _succ to the visited_stack, check the transition and accordingly updated its label
                _on_state_pattern = "\\bon\\b"

                # we also explicitly ignore "On" nodes.
                if not re.search(_on_state_pattern, _causal_succ_node):
                    self._add_transition_to_transition_system(causal_current_node=_causal_current_node,
                                                              causal_succ_node=_causal_succ_node,
                                                              game_current_node=_game_current_node,
                                                              visited_stack=visited_stack,
                                                              done_stack=done_stack)

            done_stack.append(_game_current_node)

        if plot:
            self._transition_system.plot_graph()

    def _add_transition_to_transition_system(self,
                                             causal_current_node,
                                             causal_succ_node,
                                             game_current_node,
                                             visited_stack: deque,
                                             done_stack: deque) -> None:
        """
        A helper function called by the self._build_transition_system method to add valid the edges between two states
        of the Transition System and update the label of the successor state based on the type of action being
        performed.

        The edge between two states that belong to the Transition System has the following attributes:

            1) actions = The edge action name. The name is same the one in the Causal graph
            2) weight = The weight to take that action given the actio_to_cost dictionary
        """

        # determine the action, create a valid label for the successor state and add it to successor node.
        _edge_action = self._causal_graph.causal_graph._graph[causal_current_node][causal_succ_node][0]['actions']
        _action_type: str = self._get_action_from_causal_graph_edge(_edge_action)
        _curr_node_list_lbl = self._transition_system._graph.nodes[game_current_node].get('list_ap')
        _curr_node_lbl = self._transition_system._graph.nodes[game_current_node].get('ap')

        if _action_type == "transit":
            _cost: int = self._action_to_cost.get("transit")
            if self._check_transit_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                   action=_edge_action):

                # the label does not change
                _game_succ_node = causal_succ_node + _curr_node_lbl

                if _game_succ_node not in self._transition_system._graph.nodes:
                    self._transition_system.add_state(_game_succ_node,
                                                      causal_state_name=causal_succ_node,
                                                      player="eve",
                                                      list_ap=_curr_node_list_lbl.copy(),
                                                      ap=_curr_node_lbl)

                if (game_current_node, _game_succ_node) not in self._transition_system._graph.edges:
                    self._transition_system.add_edge(game_current_node,
                                                     _game_succ_node,
                                                     actions=_edge_action,
                                                     weight=_cost)

                if _game_succ_node not in done_stack:
                    visited_stack.append(_game_succ_node)

        elif _action_type == "transfer":
            _cost: int = self._action_to_cost.get("transfer")
            if self._check_transfer_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                    action=_edge_action):

                _succ_node_list_lbl = _curr_node_list_lbl.copy()
                _, _box_loc = self._get_multiple_box_location(_edge_action)

                _succ_node_list_lbl[-1] = _box_loc[-1]
                _succ_node_lbl = self._convert_list_ap_to_str(_succ_node_list_lbl)
                _game_succ_node = causal_succ_node + _succ_node_lbl

                if _game_succ_node not in self._transition_system._graph.nodes:
                    self._transition_system.add_state(_game_succ_node,
                                                      causal_state_name=causal_succ_node,
                                                      player="eve",
                                                      list_ap=_succ_node_list_lbl.copy(),
                                                      ap=_succ_node_lbl)

                if (game_current_node, _game_succ_node) not in self._transition_system._graph.edges:
                    self._transition_system.add_edge(game_current_node,
                                                     _game_succ_node,
                                                     actions=_edge_action,
                                                     weight=_cost)

                if _game_succ_node not in done_stack:
                    visited_stack.append(_game_succ_node)

        elif _action_type == "grasp":
            _cost: int = self._action_to_cost.get("grasp")
            if self._check_grasp_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                 action=_edge_action):

                # update the corresponding box being manipulated value as "gripper" and update gripper with the
                # corresponding box id

                _succ_node_list_lbl = _curr_node_list_lbl.copy()
                _box_id, _ = self._get_box_location(_edge_action)
                _succ_node_list_lbl[_box_id] = "gripper"
                _succ_node_list_lbl[-1] = "b" + str(_box_id)

                _succ_node_lbl = self._convert_list_ap_to_str(_succ_node_list_lbl)
                _game_succ_node = causal_succ_node + _succ_node_lbl

                if _game_succ_node not in self._transition_system._graph.nodes:
                    self._transition_system.add_state(_game_succ_node,
                                                      causal_state_name=causal_succ_node,
                                                      player="eve",
                                                      list_ap=_succ_node_list_lbl.copy(),
                                                      ap=_succ_node_lbl)

                if (game_current_node, _game_succ_node) not in self._transition_system._graph.edges:
                    self._transition_system.add_edge(game_current_node,
                                                     _game_succ_node,
                                                     actions=_edge_action,
                                                     weight=_cost)

                if _game_succ_node not in done_stack:
                    visited_stack.append(_game_succ_node)

        elif _action_type == "release":
            _cost: int = self._action_to_cost.get("release")
            if self._check_release_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                   action=_edge_action):

                # update the the corresponding box_idx with the location and gripper value as "free"
                _succ_node_list_lbl = _curr_node_list_lbl.copy()
                _box_id, _box_loc = self._get_box_location(_edge_action)

                _succ_node_list_lbl[_box_id] = _box_loc
                _succ_node_list_lbl[-1] = "free"

                _succ_node_lbl = self._convert_list_ap_to_str(_succ_node_list_lbl)
                _game_succ_node = causal_succ_node + _succ_node_lbl

                if _game_succ_node not in self._transition_system._graph.nodes:
                    self._transition_system.add_state(_game_succ_node,
                                                      causal_state_name=causal_succ_node,
                                                      player="eve",
                                                      list_ap=_succ_node_list_lbl.copy(),
                                                      ap=_succ_node_lbl)

                if (game_current_node, _game_succ_node) not in self._transition_system._graph.edges:
                    self._transition_system.add_edge(game_current_node,
                                                     _game_succ_node,
                                                     actions=_edge_action,
                                                     weight=_cost)

                if _game_succ_node not in done_stack:
                    visited_stack.append(_game_succ_node)

        elif _action_type == "human-move":
            pass

        else:
            print("Looks like we encountered an invalid type of action")

    def _check_transit_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A transit action is valid when the box's current location is in line with the current configuration of the
        world.

        e.g current_node_list_lbl: ['l3', 'l4', 'l1', 'free'] - index correspond to the respective box and the value at
        that index is the box's current location in the env. Box 0 is currently in location l3 and gripper is free.

        An action "transit b0 else l3" from the current node will be a valid transition as box 0 is indeed in location
        l3 and the robot is moving from location else to l3.
        """

        # get the box id and its location
        _box_id, _box_loc = self._get_multiple_box_location(action)

        if current_node_list_lbl[_box_id] == _box_loc[-1]:
            return True

        return False

    def _check_grasp_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A grasp action is valid when the box's current location is in line with the current configuration of the
        world. An additional constraint is the gripper should be free

        e.g current_node_list_lbl: ['l3', 'l4', 'l1', 'free'] - index correspond to the respective box and the value at
        that index is the box's current location in the env. Box 0 is currently in location l3 and gripper is free.

        An action "grasp b0 l3" from the current node will be a valid action as box 0 is indeed in location l3 and the
        gripper is in "free" state
        """

        # get the box id and its location
        _box_id, _box_loc = self._get_box_location(action)

        if current_node_list_lbl[_box_id] == _box_loc:
            if current_node_list_lbl[-1] == "free":
                return True

        return False

    def _check_transfer_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A transfer action is valid when the box is currently in the grippers hand and the grippers is holding that
        particular box. Also, the box can also be transferred to a place which is does not a box already placed in it.

        e.g current_node_list_lbl: ['gripper', 'l4', 'l1', 'b0'] -  Box 0 is currently being transferred

        An action "transfer b0 l0 l2" from the current node will be a valid action as box 0 can indeed be placed in
        location l2 from location l0.
        """

        # get the box id and its location
        _box_id, _box_loc = self._get_multiple_box_location(action)

        if current_node_list_lbl[_box_id] == "gripper" and current_node_list_lbl[-1] == "b" + str(_box_id):
            if not (_box_loc[-1] in current_node_list_lbl):
                return True

        return False

    def _check_release_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A release action is valid when the box is currently in the grippers hand and the gripper is ready to drop it.
        The location where it is dropping should not be occupied by some other box

        e.g current_node_list_lbl: ['gripper', 'l4', 'l1', 'l2'] - index correspond to the respective box and the value
        at that index is the box's current location in the env. Box 0 is currently being held and the gripper is ready
        to release it in location 'l2'

        An action "release b0 l2" from the current node will be a valid action as box 0 is indeed in being manipulated
        and location 'l2' is free.
        """

        # get the box id and its location
        _box_id, _box_loc = self._get_box_location(action)

        if current_node_list_lbl[_box_id] == "gripper" and not (_box_loc in current_node_list_lbl[:-1]):
            return True

        return False

    def _get_initial_state_label(self) -> Tuple[List[str], str]:
        """
        A function that create the initial label given the grounded (True) labels in the causal graph. This is a crucial
        step because, we update the labels from the intial label.

        The code, initially, gets the grounded labels (where the boxes located at). We intialize list of the appropriate
        length.

        [0, 0, 0, ..., free] : The 0s are placeholder for box locations and the last element in the list indicates the
        state of the manipulator. Initially, the robot end effector is free.

        reuturns: A list of the form ["l1", "l2", ..., "free"], The robot's intial conf
        """

        # get the init state of the world
        _init_state_list: List[str] = list(self._causal_graph.task.initial_state)

        # initialize an empty tuple with all 0s; init_state_list has an extra free franka label that is not an "on"
        # predicate
        _init_state_label = [0 for _n in range(len(_init_state_list))]

        for _idx in range(len(_init_state_list)):
            if _idx == len(_init_state_list) - 1:
                _init_state_label[_idx] = "free"
            else:
                _init_state_label[_idx] = "0"

        for _causal_state_str in _init_state_list:
            if "on" in _causal_state_str:
                _idx, _loc_val = self._get_box_location(_causal_state_str)
                _init_state_label[_idx]: str = _loc_val
            else:
                _causal_graph_init_state: str = _causal_state_str

        return _init_state_label, _causal_graph_init_state

    def _get_box_location(self, box_location_state_str: str) -> Tuple[int, str]:
        """
        A function that returns the location of the box and the box id in the given world from a given string.
        This string could an action, state label or any other appropriate input that exactly has one box variable and
        one location vairable in the string.

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

    def _get_action_from_causal_graph_edge(self, causal_graph_edge_str: str) -> str:
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


if __name__ == "__main__":

    # define some constants
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _plotting = True

    # Define PDDL files
    domain_file_path = _project_root + "/../.." + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/../.." + "/pddl_files/blocks_world/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=_plotting)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)