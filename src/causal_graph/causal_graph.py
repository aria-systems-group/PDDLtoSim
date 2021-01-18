import os
import networkx as nx
import warnings
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
        self._raw_pddl_ts = None
        self._pddl_ltl_automata = None
        self._product = None

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
                                                  from_file=False,
                                                  pre_built=False,
                                                  save_flag=True,
                                                  debug=False,
                                                  plot=False,
                                                  human_intervention=0,
                                                  plot_raw_ts=False)

        for _u in task.facts:
            raw_transition_system.add_state(_u, player='eve', ap=_u.replace(" ", "_"))
            for _v in task.facts:
                for _action in task.operators:
                    if _u == _v:
                        continue
                    elif _u in _action.preconditions:
                        if _v in _action.add_effects or _v in _action .del_effects:
                            if (_u, _v) not in raw_transition_system._graph.edges:
                                raw_transition_system.add_edge(_u, _v,
                                                               actions=_action.name,
                                                               weight=0,
                                                               precondition=_action.preconditions,
                                                               add_effects=_action.add_effects,
                                                               del_effects=_action.del_effects)

        raw_transition_system.add_initial_states_from(task.initial_state)
        raw_transition_system.add_accepting_states_from(task.goals)
        raw_transition_system._sanity_check(debug=True)

        self._raw_pddl_ts = raw_transition_system

        if self._plot_graph:
            raw_transition_system.plot_graph()

    def _get_task(self):
        _problem = _parse(self._domain_file, self._problem_file)
        # change this in future
        self.num_of_obs = _problem.domain.predicates
        _task = _ground(_problem)

        return _task

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





    @property
    def problem_file(self):
        return self._problem_file

    @property
    def domain_file(self):
        return self._domain_file


if __name__ == "__main__":

    # define some constants
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _plotting = False

    # Define PDDL files
    domain_file_path = _project_root + "/../.." + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/../.." + "/pddl_files/blocks_world/problem.pddl"

    # Define problem and domain file, call the method for testing
    pddl_test_obj = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=_plotting)
    # build causal graph
    pddl_test_obj.build_causal_graph()

    # build the ltl automata
    pddl_test_obj.build_LTL_automato(formula="F(on_rb_l_2) & F(on_bb_1_l_0)")

    # compose the above two graphs
    pddl_test_obj.build_product()
