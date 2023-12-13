"""
 This module tests Dynamic Franka Abstraction construction and strategy synthesis.
"""

import os
import unittest

import sys
PROJECT_ROOT_1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT_1)

from typing import Set
# from parameterized import parameterized_class

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.two_player_game import TwoPlayerGame
from src.graph_construction.transition_system import FiniteTransitionSystem
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

from main import run_synthesis_and_rollout

# config flags 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_FILE_PATH = os.path.join(PROJECT_ROOT, "pddl_files/domain.pddl")
PROBLEM_FILE_PATH = os.path.join(PROJECT_ROOT, "pddl_files/problem.pddl")

LTLF_EXPECTED_DFA_OP = {
    "F(p01 | p17)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2},
    "F(p01)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 2
        },
    "F(p00)": {
        "init_state": ["q1"],
        "accp_state": ["q2"],
        "num_of_states": 1
        }
    }


LTLF_EXPECTED_CAUSAL_GRAPH_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 55,
        "num_of_edges": 204
        },
    "F(p01)": {
        "num_of_nodes": 2,
        "num_of_edges": 2
        },
    "F(p00)": {
        "num_of_nodes": 2,
        "num_of_edges": 2
        }
    }


LTLF_EXPECTED_TRANS_SYS_OP = {
    "F(p01 | p17)": {
        "num_of_nodes": 2,
        "num_of_edges": 2
        },
    "F(p01)": {
        "num_of_nodes": 2,
        "num_of_edges": 2
        },
    "F(p00)": {
        "num_of_nodes": 2,
        "num_of_edges": 2
        }
    }


LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP = {
    "F(p01 | p17)": {
        "sys_count": 2,
        "env_count": 2,
        "num_of_nodes": 2,
        "num_of_edges": 2
        },
    "F(p01)": {
        "sys_count": 2,
        "env_count": 2,
        "num_of_nodes": 2,
        "num_of_edges": 2
        },
    "F(p00)": {
        "sys_count": 2,
        "env_count": 2,
        "num_of_nodes": 2,
        "num_of_edges": 2
        }
    }

class ConstrcutGraphs():

    def __init__(self, formula: str):
        self.formula = formula
        self.causal_graph_instance = None
        self.transition_system_instance = None
        self.two_player_instance = None
        self.dfa_handle = None
        self.product_graph = None
        self.setUp()
    
    def setUp(self) -> None:
        print("**********************************************************************************************************")
        print('Formula:', self.formula)
        print("**********************************************************************************************************")
        self.causal_graph_instance = CausalGraph(problem_file=PROBLEM_FILE_PATH,
                                                domain_file=DOMAIN_FILE_PATH,
                                                draw=False)

        self.causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        # create Transition System
        self.transition_system_instance = FiniteTransitionSystem(self.causal_graph_instance)
        self.transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
        

        # create Twoplayer graph
        self.two_player_instance = TwoPlayerGame(self.causal_graph_instance, self.transition_system_instance)
        self.two_player_instance.build_two_player_game(human_intervention=2,
                                                      human_intervention_cost=0,
                                                      plot_two_player_game=False,
                                                      arch_construction=False)

        # for implicit construction, the human intervention should >=2
        self.two_player_instance.build_two_player_implicit_transition_system_from_explicit(plot_two_player_implicit_game=False)
        self.two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
        self.two_player_instance.modify_ap_w_object_types(implicit=True)

        # create DFA
        self.dfa_handle = self.two_player_instance.build_LTLf_automaton(formula=self.formula, plot=False)

        # create product graph
        self.product_graph = self.two_player_instance.build_product(dfa=self.dfa_handle,
                                                                    trans_sys=self.two_player_instance.two_player_implicit_game)
    




class TestRegretStrSynth(unittest.TestCase):
    """
     We override the setUpClass method to construct the original Two-player graph only once. 
     This class method is initialized only once when the Test class is initialized. Thus, helping us same time.
     
     Note that the order in which the various test cases will be run is determined by sorting the test function names with respect to the built-in ordering for strings.
      Link: docs.python.org/library/unittest.html

     Thus, we augement methods with numbers as test_i_<test_method_name>() where i indicates the order in which the test methods need to executed.

    """
    formula = ""
    causal_graph_instance = None
    transition_system_instance = None
    two_player_instance = None
    dfa_handle = None
    product_graph = None 

    # def __init__(self, methodName: str, formula: str):
    #     super().__init__(methodName)
    #     self.formula = formula
    
    # def __init__(self, formula: str):
    #     super().__init__()
    #     self.formula = formula

    # def __init__(self):
    #     super().__init__()
        # graph_handle = ConstrcutGraphs(formula=self.formula)
        # self.causal_graph_instance = graph_handle.causal_graph_instance
        # self.transition_system_instance = graph_handle.transition_system_instance
        # self.two_player_instance = graph_handle.two_player_instance
        # self.dfa_handle = graph_handle.dfa_handle
        # self.product_graph = graph_handle.product_graph

    # def setUp(self) -> None:
        # create causal graph
        # self.causal_graph_instance = CausalGraph(problem_file=PROBLEM_FILE_PATH,
        #                                         domain_file=DOMAIN_FILE_PATH,
        #                                         draw=False)

        # self.causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        # # create Transition System
        # self.transition_system_instance = FiniteTransitionSystem(self.causal_graph_instance)
        # self.transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
        

        # # create Twoplayer graph
        # self.two_player_instance = TwoPlayerGame(self.causal_graph_instance, self.transition_system_instance)
        # self.two_player_instance.build_two_player_game(human_intervention=2,
        #                                               human_intervention_cost=0,
        #                                               plot_two_player_game=False,
        #                                               arch_construction=False)

        # # for implicit construction, the human intervention should >=2
        # self.two_player_instance.build_two_player_implicit_transition_system_from_explicit(plot_two_player_implicit_game=False)
        # self.two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
        # self.two_player_instance.modify_ap_w_object_types(implicit=True)

        # # create DFA
        # self.dfa_handle = self.two_player_instance.build_LTL_automaton(formula=self.formula, plot=False)

        # # create product graph
        # self.product_graph = self.two_player_instance.build_product(dfa=self.dfa_handle,
        #                                                             trans_sys=self.two_player_instance.two_player_implicit_game)
        # pass
    
    # @classmethod
    # def setUpClass(cls) -> None:
    #     # create causal graph
    #     print("**********************************************************************************************************")
    #     print('Formula:', cls.formula)
    #     print("**********************************************************************************************************")
    #     cls.causal_graph_instance = CausalGraph(problem_file=PROBLEM_FILE_PATH,
    #                                             domain_file=DOMAIN_FILE_PATH,
    #                                             draw=False)

    #     cls.causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    #     # create Transition System
    #     cls.transition_system_instance = FiniteTransitionSystem(cls.causal_graph_instance)
    #     cls.transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
        

    #     # create Twoplayer graph
    #     cls.two_player_instance = TwoPlayerGame(cls.causal_graph_instance, cls.transition_system_instance)
    #     cls.two_player_instance.build_two_player_game(human_intervention=2,
    #                                                   human_intervention_cost=0,
    #                                                   plot_two_player_game=False,
    #                                                   arch_construction=False)

    #     # for implicit construction, the human intervention should >=2
    #     cls.two_player_instance.build_two_player_implicit_transition_system_from_explicit(plot_two_player_implicit_game=False)
    #     cls.two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    #     cls.two_player_instance.modify_ap_w_object_types(implicit=True)

    #     # create DFA
    #     cls.dfa_handle = cls.two_player_instance.build_LTLf_automaton(formula=cls.formula, plot=False)

    #     # create product graph
    #     cls.product_graph = cls.two_player_instance.build_product(dfa=cls.dfa_handle,
    #                                                                 trans_sys=cls.two_player_instance.two_player_implicit_game)
    
    @staticmethod
    def get_edge_weights(game) -> Set[int]:
        """
        A helper method that returns the set of all edge weights in the Minigrid Two player abstraction.
        """
        assert isinstance(game, TwoPlayerGraph), "game must be an instance of TwoPlayerGraph"
        _edge_weights: Set[int] = set({})
        for _s in  game._graph.nodes():
            for _e in  game._graph.out_edges(_s):
                if  game._graph[_e[0]][_e[1]][0].get('weight', None):
                    _edge_weights.add(game._graph[_e[0]][_e[1]][0]["weight"])

        return _edge_weights
    

    # def test_00_mySetUp(self) -> None:
    #     """
    #       This method mimics the build SetUp method except that it is called excatly once for every new formula (Test Case).
    #     """
    #     graph_handle = ConstrcutGraphs(formula=self.formula)
    #     self.causal_graph_instance = graph_handle.causal_graph_instance
    #     self.transition_system_instance = graph_handle.transition_system_instance
    #     self.two_player_instance = graph_handle.two_player_instance
    #     self.dfa_handle = graph_handle.dfa_handle
    #     self.product_graph = graph_handle.product_graph
        # self.causal_graph_instance = graph_handle.causal_graph_instance
        # self.transition_system_instance = graph_handle.transition_system_instance
        # self.two_player_instance = graph_handle.two_player_instance
        # self.dfa_handle = graph_handle.dfa_handle
        # self.product_graph = graph_handle.product_graph

    def test_01_causal_graph_construction(self):
        """
         Test PDDL to Causal Graph construction
        """
        # self.assertEqual(len(self.causal_graph_instance._causal_graph._graph.nodes()),
        #                  55,
        #                  msg=f"Mismatch in number of nodes in the Causal Graph. Should be 55 but got {len(self.causal_graph_instance._causal_graph._graph.nodes())}")

        # create the graphs
        print("Causal Graph: ")
        print(len(self.causal_graph_instance._causal_graph._graph.nodes()))
        print(len(self.causal_graph_instance._causal_graph._graph.edges()))
        # self.assertEqual(len(self.causal_graph_instance._causal_graph._graph.nodes()),
        #                  LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_nodes'],
        #                  msg=f"Mismatch in number of nodes in the Causal Graph. \
        #                       Should be {LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_nodes']} but got {len(self.causal_graph_instance._causal_graph._graph.nodes())}")
        
        # self.assertEqual(len(self.causal_graph_instance._causal_graph._graph.edges()),
        #                  LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_nodes'],
        #                  msg=f"Mismatch in number of edges in the Causal Graph. Should be 204 but got {len(self.causal_graph_instance._causal_graph._graph.edges())}")
    

    def test_02_transition_system_construction(self):
        """
         Test PDDL to Transition System Graph construction
        """
        print("Transition Graph: ")
        print(len(self.transition_system_instance.transition_system._graph.nodes()))
        print(len(self.transition_system_instance.transition_system._graph.edges()))
        # self.assertEqual(len(self.transition_system_instance.transition_system._graph.nodes()),
        #                  241,
        #                  msg=f"Mismatch in number of nodes in the Transition System. Should be 241 but got {len(self.transition_system_instance.transition_system._graph.nodes())}")
        
        # self.assertEqual(len(self.transition_system_instance.transition_system._graph.edges()),
        #                  542,
        #                  msg=f"Mismatch in number of edges in the Causal Graph. Should be 542 but got {len(self.transition_system_instance.transition_system._graph.edges())}")
    

    def test_03_two_player_graph_construction(self):
        """
         Test PDDL to Two-player Game construction with edge weights
        """
        print("Two-player Graph: ")
        # print # of Sys and Env state
        env_count = 0
        sys_count = 0
        for node, d in self.two_player_instance._two_player_implicit_game._graph.nodes(data=True):
            if d['player'] == 'adam':
                env_count += 1
            elif d['player'] == 'eve':
                sys_count += 1
            else:
                raise ValueError(f"Node {node} has no player attribute")
        
        print(sys_count)
        print(env_count)
        print(len(self.two_player_instance._two_player_implicit_game._graph.nodes()))
        print(len(self.two_player_instance._two_player_implicit_game._graph.edges()))
        
        # self.assertEqual(sys_count, 295, msg=f"Mismatch in number of system states in Two player Graph. Should be 295 but got {sys_count}")
        # self.assertEqual(env_count, 668, msg=f"Mismatch in number of Env states in Two player Graph. Should be 668 but got {env_count}")
        # self.assertEqual(len(self.two_player_instance._two_player_implicit_game._graph.nodes()), 963, msg=f"Mismatch in number of nodes in Two player Graph. Should be 963 but got {len(self.two_player_instance._two_player_implicit_game._graph.nodes())}")
        # self.assertEqual(len(self.two_player_instance._two_player_implicit_game._graph.edges()), 2180, msg=f"Mismatch in number of edges in Two player Graph. Should be 2180 but got {len(self.two_player_instance._two_player_implicit_game._graph.edges())}")
    

    def test_04_product_construction(self):
        """
         Test Product construction given Two-player Game instance and the DFA instance
        """
        print("DFA Graph: ")
        print(self.dfa_handle.init_state)
        print(self.dfa_handle.accp_states)
        print(self.dfa_handle.num_of_states)
        # self.assertEqual(EXPECTED_DFA_OP[self.formulas[0]]["init_state"], self.dfa_handle.init_state)
        # self.assertEqual(EXPECTED_DFA_OP[self.formulas[0]]["accp_state"], self.dfa_handle.accp_states)
        # self.assertEqual(EXPECTED_DFA_OP[self.formulas[0]]["num_of_states"], self.dfa_handle.num_of_states)

        print("Product Graph: ")
        print(len(self.product_graph._graph.nodes()))
        print(len(self.product_graph._graph.edges()))
        self.assertEqual(len(self.product_graph._graph.nodes()), 964, msg=f"Mismatch in number of nodes. Should be  but got {len(self.product_graph._graph.nodes())}")
        self.assertEqual(len(self.product_graph._graph.edges()), 2181 ,msg=f"Mismatch in number of edges. Should be  but got {len(self.product_graph._graph.edges())}")
    

    def test_05_min_max_value_iteration(self):
        """
         Test adversarial game playing for determining minimum energy budget 
        """
        # stop after computing the Min-max value.
        strategy_handle, _ = run_synthesis_and_rollout(strategy_type="Min-Max",
                                                       game=self.product_graph,
                                                       rollout_flag=False,
                                                       debug=True)
        self.assertTrue(strategy_handle.is_winning())
    

    def test_06_graph_of_utility_construction(self):
        """
          Test Graph of Utility (unrolling original graph) Construction 
        """
        # raise NotImplementedError()
        pass
    
    def test_07_cVal_construction(self):
        """
           Test Cooperative Value (cVal) computation. An important step in computing Best Alternative (BA) value associated with edge on Graph of Utility.
        """
        # raise NotImplementedError()
        pass
    
    def test_08_graph_of_alternatives_construction(self):
        """
          Test Graph of Best Response Construction given the Graph of Utiltiy and BA for every edge
        """
        # raise NotImplementedError()
        pass

    def test_09_reg_str_synthesis(self):
        """
         Tests Regret-minimizing strategy synthesis by playing Min-MAx game on the Graph of Best Response. 

         Note: This method only check the correctness of the code. The validity of the synthesized strategy is done by rolling out the strategy.
        """
        strategy_handle, _ = run_synthesis_and_rollout(strategy_type="Regret",
                                                       game=self.product_graph,
                                                       rollout_flag=False,
                                                       debug=True,
                                                       max_iterations=100)
        
        print("Is winning:", strategy_handle.is_winning())
        print("Reg Budget:", strategy_handle.reg_budget)
        print("Reg Factor:", strategy_handle.reg_factor)
        
        print("Min Budget (Min-Max Value - aVal on Orig. Game):", strategy_handle.min_budget)
        print("# of Nodes in Graph of Utility:", len(strategy_handle.graph_of_utility._graph.nodes()))
        print("# of Edges in Graph of Utility:", len(strategy_handle.graph_of_utility._graph.edges()))

        gou_edge_weights = TestRegretStrSynth.get_edge_weights(strategy_handle.graph_of_utility)
        print("# of leaf nodes in Graph of Utility:", len([n for n in strategy_handle.graph_of_utility.get_absorbing_state()]))
        print("Set of leaf nodes in Graph of Utility:", set([n for n in strategy_handle.graph_of_utility.get_absorbing_state()]))
        print("Edge weights in Graph of Utility:", gou_edge_weights)
        print("Best Alternative Values (cVAl):", strategy_handle.best_alternate_values)
        print("# of iterations for cVAl Computation:", strategy_handle.graph_of_utility_fp_iter)

        print("# of Nodes in Graph of Alternatives:", len(strategy_handle.graph_of_alternatives._graph.nodes()))
        print("# of Edges in Graph of Alternatives:", len(strategy_handle.graph_of_alternatives._graph.edges()))
        print("# of leaf nodes in Graph of Alternatives:", len([n for n in strategy_handle.graph_of_alternatives.get_absorbing_state()]))
        print("Set of leaf nodes in Graph of Alternatives:", set([n for n in strategy_handle.graph_of_alternatives.get_absorbing_state()]))
        print("# of iterations for reg Computation:", strategy_handle.graph_of_alternatives_fp_iter)

        print("Reg value of the game:", strategy_handle.state_values[strategy_handle.graph_of_alternatives.get_initial_states()[0][0]])
    
    def test_10_reg_str_validity(self):
        """
            Test if the optimal behaviors are in-line with the expected behaviors.
        """
        # raise NotImplementedError()
        pass


def create_suite():
    suite = unittest.TestSuite()
    for formula in LTLF_EXPECTED_DFA_OP.keys():
        # method that does the precomputation
        graph_handle = ConstrcutGraphs(formula=formula)
        
        # Added the formula and the graph handles as class attributes
        TestRegretStrSynth.formula = formula
        TestRegretStrSynth.causal_graph_instance = graph_handle.causal_graph_instance
        TestRegretStrSynth.transition_system_instance = graph_handle.transition_system_instance
        TestRegretStrSynth.two_player_instance = graph_handle.two_player_instance
        TestRegretStrSynth.dfa_handle = graph_handle.dfa_handle
        TestRegretStrSynth.product_graph = graph_handle.product_graph
        
        # create the Test suite
        suite.addTest(unittest.makeSuite(TestRegretStrSynth))
    return suite


if __name__ == "__main__":
    # suite()
    runner = unittest.TextTestRunner()
    runner.run(create_suite())
    # testing = TestRegretStrSynth()
    # testing.setUpClass()
    # # testing.test_1_causal_graph_construction()
    # testing.test_5_min_max_value_iteration()
    # testing.test_6_reg_str_synthesis()
    
    # unittest.main()