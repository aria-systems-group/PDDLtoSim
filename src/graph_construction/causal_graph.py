import os
import warnings

# import reg_syn_packages
from regret_synthesis_toolbox.src.graph import graph_factory

# import pyperplan packages
from pyperplan import _parse, _ground


class CausalGraph:
    """
    Given a problem and domain file, we would like to plot and build a "raw transition system" that only includes the
    system nodes.
    """

    def __init__(self, problem_file: str, domain_file: str, draw: bool = False):
        self._problem_file = problem_file
        self._domain_file = domain_file
        self._plot_graph = draw
        self._causal_graph = None
        self._task = None
        self._problem = None
        self._task_objects: list = []
        self._task_intervening_locations: list = []
        self._task_non_intervening_locations: list = []
        self._task_locations: list = []
        self._get_task_and_problem()
        self._get_boxes_and_location_from_problem()

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

    @property
    def task_objects(self):
        return self._task_objects

    @property
    def task_locations(self):
        return self._task_locations

    @property
    def task_intervening_locations(self):
        return self._task_intervening_locations

    @property
    def task_non_intervening_locations(self):
        return self._task_non_intervening_locations

    @property
    def causal_graph(self):
        return self._causal_graph

    def get_task_name(self):
        return self._task.name

    def build_causal_graph(self, add_cooccuring_edges: bool = False, debug: bool = False):
        """
        A method that gets the task, dumps the respective data in a yaml file and build a graph using the
        regret_synthesis_toolbox graph factory which reads the dumped yaml file.
        """
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
        raw_transition_system._sanity_check(debug=debug)

        self._causal_graph = raw_transition_system

        if self._plot_graph:
            raw_transition_system.plot_graph()

    def _get_task_and_problem(self):
        self._problem = _parse(self._domain_file, self._problem_file)
        # change this in future
        self.num_of_obs = self._problem.domain.predicates
        self._task = _ground(self._problem)

    def _get_boxes_and_location_from_problem(self):
        """
        A helper function to return the boxes and location associated with a problem in a given domain.
        """
        # get objects and location from the problem instance
        for _object, _type in self._problem.objects.items():
            if _type.name == 'box':
                self._task_objects.append(_object)

            if _type.name == 'box_loc':
                self._task_non_intervening_locations.append(_object)
                self._task_locations.append(_object)

            if _type.name == 'hbox_loc':
                self._task_intervening_locations.append(_object)
                self._task_locations.append(_object)


if __name__ == "__main__":

    # define some constants
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _plotting = False

    # Define PDDL files
    domain_file_path = _project_root + "/../.." + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    problem_file_ath = _project_root + "/../.." + "/pddl_files/two_table_scenario/diagonal/problem.pddl"
    # domain_file_path = _project_root + "/../.." + "/pddl_files/blocks_world/domain.pddl"
    # problem_file_ath = _project_root + "/../.." + "/pddl_files/blocks_world/problem.pddl"

    # Define problem and domain file, call the method for testing
    pddl_test_obj = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=_plotting)

    # build causal graph
    pddl_test_obj.build_causal_graph(add_cooccuring_edges=False, debug=False)
