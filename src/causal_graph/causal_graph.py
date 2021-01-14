import os
import networkx as nx
import pygraphviz as pgv

from matplotlib import pyplot as plt

# import reg_syn_packages

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

    def build_causal_graph(self):
        """
        A method that gets the task, dumps the respective data in a yaml file and build a graph using the
        regret_synthesis_toolbox graph factory which reads the dumped yaml file.
        """
        pddl_graph = pgv.AGraph(directed=True, size=1000)
        # graph.node_attr['style']='filled'
        pddl_graph.node_attr['shape'] = 'box'
        pddl_graph.node_attr['fixedsize'] = 'false'
        pddl_graph.node_attr['fontsize'] = 16
        pddl_graph.node_attr['fontcolor'] = '#000000'
        pddl_graph.node_attr['width'] = 1.5

        task = self._get_task()

        for u in task.facts:
            for v in task.facts:
                for op in task.operators:
                    if u == v:
                        continue
                    elif u in op.preconditions:
                        if v in op.add_effects or v in op.del_effects:
                            try:
                                pddl_graph.get_node(u)
                            except KeyError:
                                pddl_graph.add_node(u)
                            try:
                                pddl_graph.get_node(v)
                            except KeyError:
                                pddl_graph.add_node(v)
                            pddl_graph.add_edge(u, v, minlen=5, xlabel=op.name)

        if self._plot_graph:
            # plt.subplot(111)
            # nx.draw(pddl_graph, with_labels=True)
            # plt.show()
            pddl_graph.write("pddl_graph.dot")
            pddl_graph.draw("pddl_graph.pdf", prog='dot')

    def _get_task(self):
        _problem = _parse(self._domain_file, self._problem_file)
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
