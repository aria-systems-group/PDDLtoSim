import os
import networkx as nx
import yaml
import pygraphviz as pgv

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

    def _dump_pddl_to_yaml(self, task, file_name: str):
        """
        A build in method that dumps a given Task object into a yaml file that can then be used to build an
        Transition System using the regret_synthesis_toolbox graph factory.
        """
        #create an nx object as its easier to dump it
        # graph = nx.DiGraph()
        #
        # for u in task.facts:
        #     # add init node flag if it is the initial state
        #     if u in task.initial_state:
        #         graph.add_node(u, init=True)
        #     for v in task.facts:
        #         for op in task.operators:
        #             if u == v:
        #                 continue
        #             elif u in op.preconditions:
        #                 if v in op.add_effects or v in op.del_effects:
        #                     if (u, v) not in graph.edges:
        #                         graph.add_edge(u, v, action=set([op.name]))
        #                     else:
        #                         graph.edges[u, v]["action"].add(op.name)
        # _nodes = [_n for _n in graph.nodes.data()]
        # _edges = [_e for _e in graph.edges.data()]

        # store all the respective
        _start_state = set(task.initial_state)
        _goal_state = set(task.goals)
        _num_of_obs = self.num_of_obs

        # create node dictionary
        _nodes = []
        for _n in task.facts:
            if _n in task.initial_state:
                _nodes.append((_n, {'init': True}))
            else:
                _nodes.append((_n, {}))

        _edges = [_e.name for _e in task.operators]
        _alphabet_size = len(_edges)
        _num_states = len(_nodes)

        # create a dict and dump it using yaml's dump functionality
        _graph_dict = dict(
            alphabet_size=_alphabet_size,
            num_states=_num_states,
            num_obs=_num_of_obs,
            start_state=_start_state,
            nodes=_nodes,
            edges=_edges
        )

        try:
            with open(file_name, 'w') as outfile:
                yaml.dump(_graph_dict, outfile, default_flow_style=False)

        except FileNotFoundError:
            print(FileNotFoundError)
            print(f"The file {file_name} could not be found."
                  f" This could be because I could not find the folder to dump in")

    def build_causal_graph(self):
        """
        A method that gets the task, dumps the respective data in a yaml file and build a graph using the
        regret_synthesis_toolbox graph factory which reads the dumped yaml file.
        """

        task = self._get_task()

        config_yaml = "/config/" + task.name

        raw_transition_system = graph_factory.get('TS',
                                                  raw_trans_sys=None,
                                                  config_yaml=config_yaml,
                                                  graph_name=task.name,
                                                  from_file=True,
                                                  pre_built=False,
                                                  save_flag=True,
                                                  debug=False,
                                                  plot=True,
                                                  human_intervention=0,
                                                  plot_raw_ts=True)

        for _u in task.facts:
            raw_transition_system.add_state(_u, player='eve')
            for _v in task.facts:
                for _action in task.operators:
                    if _u == _v:
                        continue
                    elif _u in _action.preconditions:
                        if _v in _action.add_effects or _v in _action .del_effects:
                            if (_u, _v) not in raw_transition_system._graph.edges:
                                raw_transition_system.add_edge(_u, _v,
                                                               actions=_action.name,
                                                               precondition=_action.preconditions,
                                                               add_effects=_action.add_effects,
                                                               del_effects=_action.del_effects)

        raw_transition_system.add_initial_states_from(task.initial_state)
        raw_transition_system.add_accepting_states_from(task.goals)

        raw_transition_system.plot_graph()

    def _get_task(self):
        _problem = _parse(self._domain_file, self._problem_file)
        # change this in future
        self.num_of_obs = _problem.domain.predicates
        _task = _ground(_problem)

        return _task

    @property
    def problem_file(self):
        return self._problem_file

    @property
    def domain_file(self):
        return self._domain_file


if __name__ == "__main__":

    # define some constants
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _plotting = True

    # Define PDDL files
    domain_file_path = _project_root + "/../.." + "/pddl_files/domain.pddl"
    problem_file_ath = _project_root + "/../.." + "/pddl_files/problem.pddl"

    # Define problem and domain file, call the method for testing
    pddl_test_obj = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=_plotting)
    pddl_test_obj.build_causal_graph()
