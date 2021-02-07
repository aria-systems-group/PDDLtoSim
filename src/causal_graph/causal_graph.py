import os
import warnings
import re

from typing import Tuple, Optional, Dict, Union
from collections import deque

# import reg_syn_packages
from regret_synthesis_toolbox.src.graph import graph_factory
from regret_synthesis_toolbox.src.graph import FiniteTransSys

# import pyperplan packages
from pyperplan import _parse, _ground

"""
Lets write a class that build a causal graph from the given PDDL problem and domain files. And from this graph we try
simulating it in a simulation env of our choice 
"""


class CausalGraph:
    """
    Given a problem and domain file, we would like to plot and build a "raw transition system" that only includes the
    system nodes.
    """

    def __init__(self, problem_file: str, domain_file: str, draw: bool = False):
        self._problem_file = problem_file
        self._domain_file = domain_file
        self._plot_graph = draw
        self._raw_pddl_ts = None
        self._pddl_ltl_automata = None
        self._product = None
        self._task = None
        self._problem = None
        self._get_task_and_problem()

        @property
        def problem_file(self):
            return self._problem_file

        @property
        def domain_file(self):
            return self._domain_file

        @property
        def task(self):
            return self._task

        @property
        def problem(self):
            return self._problem

    def _build_two_player_graph_from_causal_raph(self, raw_transition_system:  FiniteTransSys) -> FiniteTransSys:
        """
        A method that builds an explicit two player graph for a transition that only has system nodes and actions that
        belong to robot and human as well.

        @param: raw_transition_system: The transition system that dictates how the system states evolve in presence of
        human-intervention
        """

        # for every transition from a system node, add a intermediate human state
        pass

    def _get_valid_set_of_ap(self, objects, locations: list, print_valid_labels: bool = False):
        """
        A function that given a list of objects and location creates a super-set of atomic proposition. This superset
        will be used to verify the validity of a given atomic proposition.

        Given: say 3 objects (say b0, b1, b2) and 3 locations l0, l1, l2 we create a tuple
        {<box_0_location>, <box_1_location>, <box_2_location>, <gripper_status>}

        The number of box location depends on the # of objects - index corresponds to the box and the value is its loc.
        box_location can take values -
                'loc#' - all possible locations
                'gripper' - if it is being held

        gripper_status can take values -
                'box#'- holding the box
                'loc#'- is it ready to drop it
                'free'- otherwise
        """

        # number of possible objects it can hold + possible location it can drop + free states
        _valid_gripper_states = objects + locations + ['free']

        _num_of_objects = len(objects)

        _valid_labels = set()

        # Sample ap is : {[gripper, lo, l1, l2], [gripper, lo, l1, l2], [gripper, l0, l1, l2],
        # [free, l0, l1, l3, b0, b1, b3]}

        _valid_box_values = ['gripper'] + locations
        pass

        # NOTE: Need to complete this implementation

        # def loop_rec(_ap, _tuple_idx):
        #     if _tuple_idx > 1:
        #         for _box_state in _valid_box_values:
        #             loop_rec(_ap, _tuple_idx - 1)
        #             for _l in _ap:
        #                 _l.append(_box_state)
        #     else:
        #         for _gripper_state in _valid_gripper_states:
        #             _ap.append([_gripper_state])
        #         return _ap

        # loop_rec([], 5)

        for _l1 in _valid_box_values:
            for _l2 in _valid_box_values:
                for _l3 in _valid_box_values:
                    for _l4 in _valid_box_values:
                        for _l5 in _valid_box_values:
                            for _gripper in _valid_gripper_states:
                                _ap = tuple([_l1, _l2, _l3, _l4, _l5, _gripper])
                                _valid_labels.add(_ap)

        if print_valid_labels:
            print(_valid_labels)

    def _get_box_location(self, box_location_state_str: str) -> Tuple[int, str]:
        """
        A function that returns the location of the box and the box id in the given world.

        e.g Str: on b# l1 then return l1

        NOTE: The string should be exactly in the above formation i.e on<whitespace>b#<whitespave>l#. We can swap
         between small and capital i.e 'l' & 'L' are valid.
        """
        _loc_pattern = "[l|L]\d*"
        try:
            _loc_state: str = re.search(_loc_pattern, box_location_state_str).group()
        except AttributeError:
            print(f"The causal_state_string {box_location_state_str} dose not contain location of the box")

        _box_pattern = "[b|B]\d*"
        try:
            _box_state: str = re.search(_box_pattern, box_location_state_str).group()
        except AttributeError:
            print(f"The causal_state_string {box_location_state_str} dose not contain box id")

        _box_id_pattern = "\d+"
        _box_id: int = int(re.search(_box_id_pattern, _box_state).group())

        return _box_id, _loc_state

    def _get_action_from_causal_graph_edge(self, causal_graph_edge_str: str) -> str:
        """
        A function to extract the action type given an edge string (a valid action) on the causal graph. Currently the
        valid action types are:

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

    def _check_transit_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A transit action is valid when the box's current location is in line with the current configuration of the
        world.

        e.g current_node_list_lbl: ['l3', 'l4', 'l1', 'free'] - index correspond to the respective box and the value at
        that index is the box's current location in the env. Box 0 is currently in location l3 and gripper is free.

        An action "transit b0 l3" from the current node will be a valid transition as box 0 is indeed in location l3
        """

        # get the box id and its location
        _box_id, _box_loc = self._get_box_location(action)

        if current_node_list_lbl[_box_id] == _box_loc:
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
        A grasp action is valid when the box's current location is in line with the current configuration of the
        world. An additional constraint is the gripper should be free

        e.g current_node_list_lbl: ['l3', 'l4', 'l1', 'free'] - index correspond to the respective box and the value at
        that index is the box's current location in the env. Box 0 is currently in location l3 and gripper is free.

        An action "grasp b0 l3" from the current node will be a valid action as box 0 is indeed in location l3 and the
        gripper is in "free" state
        """

        return False

    def _check_release_action_validity(self, current_node_list_lbl: list, action: str) -> bool:
        """
        A grasp action is valid when the box's current location is in line with the current configuration of the
        world. An additional constraint is the gripper should be free

        e.g current_node_list_lbl: ['l3', 'l4', 'l1', 'free'] - index correspond to the respective box and the value at
        that index is the box's current location in the env. Box 0 is currently in location l3 and gripper is free.

        An action "grasp b0 l3" from the current node will be a valid action as box 0 is indeed in location l3 and the
        gripper is in "free" state
        """

        return False

    def _add_transition_to_single_player_pddl_ts(self,
                                                 causal_current_node,
                                                 causal_succ_node,
                                                 game_current_node,
                                                 visitedstack: deque) -> None:
        """
        A helped function that called by the self._build_transition_system method to add the edge between two states and
        update the label of the successor state based on the type of action being performed.
        """

        # determine the action, create a valid label for the successor state and add it to successor node.
        _edge_action = self._raw_pddl_ts._graph[causal_current_node][causal_succ_node][0]['actions']
        _action_type: str = self._get_action_from_causal_graph_edge(_edge_action)
        if _action_type == "transit":
            _curr_node_list_lbl = self.single_player_pddl_ts._graph.nodes[game_current_node].get('list_ap')
            _curr_node_lbl = self.single_player_pddl_ts._graph.nodes[game_current_node].get('ap')

            # we need to check the validity of this action
            if self._check_transit_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                   action=_edge_action):

                # the label does not change
                _game_succ_node = causal_succ_node + _curr_node_lbl

                self.single_player_pddl_ts.add_state(_game_succ_node,
                                                     causal_state_name=causal_succ_node,
                                                     player="eve",
                                                     list_ap=_curr_node_list_lbl.copy(),
                                                     ap=_curr_node_lbl)

                self.single_player_pddl_ts.add_edge(game_current_node,
                                                    _game_succ_node,
                                                    action=_edge_action)

                visitedstack.append(_game_succ_node)


        elif _action_type == "transfer":
            pass

        elif _action_type == "grasp":
            # we need to check the validity of the transition
            _curr_node_list_lbl = self.single_player_pddl_ts._graph.nodes[game_current_node].get('list_ap')
            _curr_node_lbl = self.single_player_pddl_ts._graph.nodes[game_current_node].get('ap')

            # we need to check the validity of this transition
            if self._check_grasp_action_validity(current_node_list_lbl=_curr_node_list_lbl,
                                                 action=_edge_action):

                # update the corresponding box being manipulated value as "gripper" and update gripper with the
                # corresponding box id

                _succ_node_list_lbl = _curr_node_list_lbl
                _box_id, _ = self._get_box_location(_edge_action)
                _succ_node_list_lbl[_box_id] = "gripper"
                _succ_node_list_lbl[-1] = "b" + str(_box_id)

                _succ_node_lbl = self._convert_list_ap_to_str(_succ_node_list_lbl)
                _game_succ_node = causal_succ_node + _succ_node_lbl

                self.single_player_pddl_ts.add_state(_game_succ_node,
                                                     causal_state_name=causal_succ_node,
                                                     player="eve",
                                                     list_ap=_succ_node_list_lbl.copy(),
                                                     ap=_succ_node_lbl)

                self.single_player_pddl_ts.add_edge(game_current_node,
                                                    _game_succ_node,
                                                    action=_edge_action)

                visitedstack.append(_game_succ_node)

        elif _action_type == "release":
            pass

        elif _action_type == "human-move":
            pass

        else:
            print("Looks like we encountered an invalid type of action")

        return

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

    def _build_transition_system(self):
        """
        A function that build the transition system. This function initially build the causal graph. We then iterate
        through the states of the graph, starting from the initial state, and updating the label(atomic proposition)
        for each state till we label all the states in the graph.
        """

        self.build_causal_graph(add_cooccuring_edges=False)

        # get the init state of the world
        _init_state_list = list(self._task.initial_state)

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
                _init_state_label[_idx] = _loc_val
            else:
                _causal_graph_init_state = _causal_state_str

        # lets have two stack - visitedStack and doneStack
        # As you encounter nodes, keep adding them to the visitedStack. As you encounter a neighbour that you already
        # visited, pop that node and add that node to the done stack. Repeat the process till the visitedStack is empty.


        visited_stack = deque()
        done_stack = deque()

        _config_yaml = "/config/" + "raw_single_player" + self._task.name

        self.single_player_pddl_ts = graph_factory.get('TS',
                                                       raw_trans_sys=None,
                                                       config_yaml=_config_yaml,
                                                       graph_name="raw_single_player" + self._task.name,
                                                       from_file=False,
                                                       pre_built=False,
                                                       save_flag=True,
                                                       debug=False,
                                                       plot=False,
                                                       human_intervention=0,
                                                       plot_raw_ts=False)

        _causal_current_node = _causal_graph_init_state
        _str_curr_lbl = self._convert_list_ap_to_str(_init_state_label)

        _game_curr_node = _causal_current_node + _str_curr_lbl

        visited_stack.append(_game_curr_node)

        self.single_player_pddl_ts.add_state(_game_curr_node,
                                             causal_state_name=_causal_current_node,
                                             player="eve",
                                             list_ap=_init_state_label.copy(),
                                             ap=_str_curr_lbl)

        while visited_stack:
            # in the initial run we skip this intial pop and proceed directly to the successes
            # if len(visited_stack) > 1:

            _game_current_node = visited_stack.popleft()
            # _curr_node_list_lbl = self.single_player_pddl_ts._graph.nodes[_game_current_node].get('list_ap')
            # _curr_node_lbl = self.single_player_pddl_ts._graph.nodes[_game_current_node].get('ap')
            _causal_current_node = self.single_player_pddl_ts._graph.nodes[_game_current_node].get('causal_state_name')
            # _game_curr_node = _causal_current_node + _curr_node_lbl
            # self.single_player_pddl_ts.add_state(_game_curr_node,
            #                                      causal_state_name=_causal_current_node,
            #                                      player="eve",
            #                                      list_ap=_curr_node_list_lbl,
            #                                      ap=_curr_node_lbl)

            for _causal_succ_node in self._raw_pddl_ts._graph[_causal_current_node]:
                # add _succ to the visited_stack, check the transition and accordingly updated its label
                # visited_stack.append(_causal_succ_node)

                self._add_transition_to_single_player_pddl_ts(causal_current_node=_causal_current_node,
                                                              causal_succ_node=_causal_succ_node,
                                                              game_current_node=_game_current_node,
                                                              visitedstack=visited_stack)


        # _init_state = _causal_graph_init_state
        # for _v in self._raw_pddl_ts._graph[_causal_graph_init_state]:
        #     # now if the outgoing edge is transit then no change in the label.
        #     _edge_action = self._raw_pddl_ts._graph[_init_state][_v][0]['actions']
        #     _action_type: str = self._get_action_from_causal_graph_edge(_edge_action)
        #     if _action_type == "transit":
        #         pass
        #
        #     elif _action_type == "transfer":
        #         pass
        #
        #     elif _action_type == "grasp":
        #         pass
        #
        #     elif _action_type == "release":
        #         pass
        #
        #     elif _action_type == "human-move":
        #         pass
        #
        #     else:
        #         print("Looks like we encountered an invalid type of action")

    def build_causal_graph(self, add_cooccuring_edges: bool = False):
        """
        A method that gets the task, dumps the respective data in a yaml file and build a graph using the
        regret_synthesis_toolbox graph factory which reads the dumped yaml file.
        """

        _boxes, _locations = self._get_boxes_and_location_from_problem(self._problem)

        self._get_valid_set_of_ap(objects=_boxes,
                                  locations=_locations,
                                  print_valid_labels=True)

        config_yaml = "/config/" + self._task.name

        raw_transition_system = graph_factory.get('TS',
                                                  raw_trans_sys=None,
                                                  config_yaml=config_yaml,
                                                  graph_name=self._task.name,
                                                  from_file=False,
                                                  pre_built=False,
                                                  save_flag=True,
                                                  debug=False,
                                                  plot=False,
                                                  human_intervention=0,
                                                  plot_raw_ts=False)

        # based on this graph we need to construct a graph in which each system state consists of :
        # L(boxi), L(gripper)

        for _u in self._task.facts:
            raw_transition_system.add_state(_u, player='eve', ap=_u.replace(" ", "_"))
            for _v in self._task.facts:
                for _action in self._task.operators:
                    if _u == _v:
                        continue
                    elif _u in _action.preconditions:
                        if _v in _action.add_effects:
                            if (_u, _v) not in raw_transition_system._graph.edges:
                                raw_transition_system.add_edge(_u, _v,
                                                               actions=_action.name,
                                                               weight=0,
                                                               precondition=_action.preconditions,
                                                               add_effects=_action.add_effects,
                                                               del_effects=_action.del_effects)

                            elif add_cooccuring_edges:
                                raw_transition_system.add_edge(_u, _v,
                                                               actions=_action.name,
                                                               weight=0,
                                                               precondition=_action.preconditions,
                                                               add_effects=_action.add_effects,
                                                               del_effects=_action.del_effects)

        raw_transition_system.add_initial_states_from(self._task.initial_state)
        raw_transition_system.add_accepting_states_from(self._task.goals)
        raw_transition_system._sanity_check(debug=True)

        self._raw_pddl_ts = raw_transition_system

        if self._plot_graph:
            raw_transition_system.plot_graph()

    def _get_task_and_problem(self):
        self._problem = _parse(self._domain_file, self._problem_file)
        # change this in future
        self.num_of_obs = self._problem.domain.predicates
        self._task = _ground(self._problem)

    def _get_boxes_and_location_from_problem(self, problem):
        """
        A helper function to return the boxes and location associated with a problem in a given domain.
        """
        # get objects and location from the problem instance
        _objects = []
        _locations = []

        for _object, _type in problem.objects.items():
            if _type.name == 'location':
                _objects.append(_object)

            if _type.name == 'box':
                _locations.append(_object)

        return _objects, _locations

    def build_LTL_automato(self, formula: str, debug: bool=False):
        """
        A method to construct automata using the regret_synthesis_tool.
        """

        if not isinstance(formula, str):
            warnings.warn("Please make sure the input formula is of type string.")

        _ltl_automata = graph_factory.get('DFA',
                                          graph_name="pddl_ltl",
                                          config_yaml="/config/pddl_ltl",
                                          save_flag=True,
                                          sc_ltl=formula,
                                          use_alias=False,
                                          plot=False)

        self._pddl_ltl_automata =_ltl_automata

        if debug:
            print(f"The pddl formula is : {formula}")

    def build_product(self):
        _product_automaton = graph_factory.get("ProductGraph",
                                               graph_name="pddl_product_graph",
                                               config_yaml="/config/pddl_product_graph",
                                               trans_sys=self._raw_pddl_ts,
                                               dfa=self._pddl_ltl_automata,
                                               save_flag=True,
                                               prune=False,
                                               debug=False,
                                               absorbing=True,
                                               finite=False,
                                               plot=True)

        print("interesting")

if __name__ == "__main__":

    # define some constants
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _plotting = False

    # Define PDDL files
    domain_file_path = _project_root + "/../.." + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/../.." + "/pddl_files/blocks_world/problem.pddl"

    # Define problem and domain file, call the method for testing
    pddl_test_obj = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=_plotting)

    pddl_test_obj._build_transition_system()

    # build causal graph
    # pddl_test_obj.build_causal_graph()

    # build the ltl automata
    pddl_test_obj.build_LTL_automato(formula="F(on_rb_l_2) & F(on_bb_1_l_0)")

    # compose the above two graphs
    pddl_test_obj.build_product()
