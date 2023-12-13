"""
 This module tests Dynamic Franka Abstraction construction and strategy synthesis.
"""

import os
import unittest

from typing import Set

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.two_player_game import TwoPlayerGame
from src.graph_construction.transition_system import FiniteTransitionSystem
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

from main import run_synthesis_and_rollout
from tests.test_regret_str_synth.expected_ops import LTLF_EXPECTED_DFA_OP, LTLF_EXPECTED_CAUSAL_GRAPH_OP, \
    LTLF_EXPECTED_TRANS_SYS_OP, LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP, LTLF_EXPECTED_PRODUCT_GRAPH_OP, LTLF_EXPECTED_MIN_MAX_VALUE_OP, \
    LTLF_EXPECTED_GoU_OP, LTLF_EXPECTED_CVAL_OP, LTLF_EXPECTED_GoAlt_OP, LTLF_EXPECTED_REG_STR_OP

# config flags 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_FILE_PATH = os.path.join(PROJECT_ROOT, "pddl_files/domain.pddl")
PROBLEM_FILE_PATH = os.path.join(PROJECT_ROOT, "pddl_files/problem.pddl")


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
    strategy_handle = None
    rollout_handle = None
    

    def test_01_causal_graph_construction(self):
        """
         Test PDDL to Causal Graph construction
        """
        # create the causal graph
        print("Testing Causal Graph: ")
        self.assertEqual(len(self.causal_graph_instance._causal_graph._graph.nodes()),
                         LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_nodes'],
                         msg=f"Mismatch in number of nodes in the Causal Graph. \
                              Should be {LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_nodes']} but got {len(self.causal_graph_instance._causal_graph._graph.nodes())}")
        
        self.assertEqual(len(self.causal_graph_instance._causal_graph._graph.edges()),
                         LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_edges'],
                         msg=f"Mismatch in number of edges in the Causal Graph. Should be {LTLF_EXPECTED_CAUSAL_GRAPH_OP[self.formula]['num_of_edges']} but got {len(self.causal_graph_instance._causal_graph._graph.edges())}")
    

    def test_02_transition_system_construction(self):
        """
         Test PDDL to Transition System Graph construction
        """
        print("Testing Transition Graph: ")
        self.assertEqual(len(self.transition_system_instance.transition_system._graph.nodes()),
                         LTLF_EXPECTED_TRANS_SYS_OP[self.formula]['num_of_nodes'],
                         msg=f"Mismatch in number of nodes in the Transition System. Should be {LTLF_EXPECTED_TRANS_SYS_OP[self.formula]['num_of_nodes']} but got {len(self.transition_system_instance.transition_system._graph.nodes())}")
        
        self.assertEqual(len(self.transition_system_instance.transition_system._graph.edges()),
                         LTLF_EXPECTED_TRANS_SYS_OP[self.formula]['num_of_edges'],
                         msg=f"Mismatch in number of edges in the Causal Graph. Should be {LTLF_EXPECTED_TRANS_SYS_OP[self.formula]['num_of_edges']} but got {len(self.transition_system_instance.transition_system._graph.edges())}")
    

    def test_03_two_player_graph_construction(self):
        """
         Test PDDL to Two-player Game construction with edge weights
        """
        print("Testing Two-player Graph: ")
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
        
        self.assertEqual(sys_count,
                         LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP[self.formula]["sys_count"],
                         msg=f"Mismatch in number of system states in Two player Graph. Should be 295 but got {sys_count}")
        self.assertEqual(env_count, LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP[self.formula]["env_count"], msg=f"Mismatch in number of Env states in Two player Graph. Should be 668 but got {env_count}")
        self.assertEqual(len(self.two_player_instance._two_player_implicit_game._graph.nodes()), LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP[self.formula]["num_of_nodes"], msg=f"Mismatch in number of nodes in Two player Graph. Should be 963 but got {len(self.two_player_instance._two_player_implicit_game._graph.nodes())}")
        self.assertEqual(len(self.two_player_instance._two_player_implicit_game._graph.edges()), LTLF_EXPECTED_TWO_PLAYER_GRAPH_OP[self.formula]["num_of_edges"], msg=f"Mismatch in number of edges in Two player Graph. Should be 2180 but got {len(self.two_player_instance._two_player_implicit_game._graph.edges())}")
    

    def test_04_product_construction(self):
        """
         Test Product construction given Two-player Game instance and the DFA instance
        """
        print("Testing DFA Graph: ")
        self.assertEqual(LTLF_EXPECTED_DFA_OP[self.formula]["init_state"], self.dfa_handle.init_state)
        self.assertEqual(LTLF_EXPECTED_DFA_OP[self.formula]["accp_state"], self.dfa_handle.accp_states)
        self.assertEqual(LTLF_EXPECTED_DFA_OP[self.formula]["num_of_states"], self.dfa_handle.num_of_states)

        print("Testing Product Graph: ")
        self.assertEqual(len(self.product_graph._graph.nodes()), LTLF_EXPECTED_PRODUCT_GRAPH_OP[self.formula]["num_of_nodes"], msg=f"Mismatch in number of nodes. Should be {LTLF_EXPECTED_PRODUCT_GRAPH_OP[self.formula]['num_of_nodes']} but got {len(self.product_graph._graph.nodes())}")
        self.assertEqual(len(self.product_graph._graph.edges()), LTLF_EXPECTED_PRODUCT_GRAPH_OP[self.formula]["num_of_edges"] ,msg=f"Mismatch in number of edges. Should be {LTLF_EXPECTED_PRODUCT_GRAPH_OP[self.formula]['num_of_edges']} but got {len(self.product_graph._graph.edges())}")
    

    def test_05_min_max_value_iteration(self):
        """
         Test adversarial game playing for determining minimum energy budget 
        """
        # stop after computing the Min-max value.
        print("Min-Max on Orig Game")
        self.assertEqual(self.strategy_handle.min_budget, LTLF_EXPECTED_MIN_MAX_VALUE_OP[self.formula]["min_max value"], msg=f"Mismatch in Min-Max value. Should be {LTLF_EXPECTED_MIN_MAX_VALUE_OP[self.formula]['min_max value']} but got {self.strategy_handle.min_budget}")
        self.assertTrue(self.strategy_handle.reg_budget >= self.strategy_handle.min_budget, msg="Regret Budget should be greater than or equal to Min-Max value")
        self.assertEqual(self.strategy_handle.is_winning(), LTLF_EXPECTED_MIN_MAX_VALUE_OP[self.formula]["is_winning"], msg=f"Mismatch in is_winning(). Should be {LTLF_EXPECTED_MIN_MAX_VALUE_OP[self.formula]['is_winning']} but got {self.strategy_handle.is_winning()}")
    

    def test_06_graph_of_utility_construction(self):
        """
         Test Graph of Utility (unrolling original graph) Construction 
        """
        print("Testing Graph of Utility (unrolling original graph) Construction")
        self.assertEqual(len(self.strategy_handle.graph_of_utility._graph.nodes()), LTLF_EXPECTED_GoU_OP[self.formula]["num_of_nodes"], msg=f"Mismatch in number of nodes in Graph of Utility. Should be {LTLF_EXPECTED_GoU_OP[self.formula]['num_of_nodes']} but got {len(self.strategy_handle.graph_of_utility._graph.nodes())}")
        self.assertEqual(len(self.strategy_handle.graph_of_utility._graph.edges()), LTLF_EXPECTED_GoU_OP[self.formula]["num_of_edges"], msg=f"Mismatch in number of edges in Graph of Utility. Should be {LTLF_EXPECTED_GoU_OP[self.formula]['num_of_edges']} but got {len(self.strategy_handle.graph_of_utility._graph.edges())}")

        gou_edge_weights = ConstrcutGraphs.get_edge_weights(self.strategy_handle.graph_of_utility)
        num_of_leaf_nodes = len([n for n in self.strategy_handle.graph_of_utility.get_absorbing_state()])
        set_of_leaf_nodes = set([n for n in self.strategy_handle.graph_of_utility.get_absorbing_state()])
        self.assertEqual(num_of_leaf_nodes, LTLF_EXPECTED_GoU_OP[self.formula]["num_of_leaf_nodes"], msg=f"Mismatch in number of leaf nodes in Graph of Utility. Should be {LTLF_EXPECTED_GoU_OP[self.formula]['num_of_leaf_nodes']} but got {num_of_leaf_nodes}")
        self.assertEqual(set_of_leaf_nodes, LTLF_EXPECTED_GoU_OP[self.formula]["leaf_nodes"], msg=f"Mismatch in set of leaf nodes in Graph of Utility. Should be {LTLF_EXPECTED_GoU_OP[self.formula]['leaf_nodes']} but got {set_of_leaf_nodes}")
        self.assertEqual(gou_edge_weights, LTLF_EXPECTED_GoU_OP[self.formula]["edge_weights"], msg=f"Mismatch in edge weights in Graph of Utility. Should be {LTLF_EXPECTED_GoU_OP[self.formula]['edge_weights']} but got {gou_edge_weights}")


    def test_07_cVal_construction(self):
        """
         Test Cooperative Value (cVal) computation. An important step in computing Best Alternative (BA) value associated with edge on Graph of Utility.
        """
        print("Testing Cooperative Value (cVal) computation")
        self.assertEqual(self.strategy_handle.best_alternate_values, LTLF_EXPECTED_CVAL_OP[self.formula]["cval"], msg=f"Mismatch in Best Alternative Values (cVal). Should be {LTLF_EXPECTED_CVAL_OP[self.formula]['cval']} but got {self.strategy_handle.best_alternate_values}")
        self.assertEqual(self.strategy_handle.graph_of_utility_fp_iter, LTLF_EXPECTED_CVAL_OP[self.formula]["iters"], msg=f"Mismatch in # of iterations for cVal Computation. Should be {LTLF_EXPECTED_CVAL_OP[self.formula]['iters']} but got {self.strategy_handle.graph_of_utility_fp_iter}")
    
    def test_08_graph_of_alternatives_construction(self):
        """
          Test Graph of Best Response Construction given the Graph of Utiltiy and BA for every edge
        """
        print("Testing Graph of Best Response Construction")
        num_of_nodes = len(self.strategy_handle.graph_of_alternatives._graph.nodes())
        num_of_edges = len(self.strategy_handle.graph_of_alternatives._graph.edges())
        num_of_leaf_nodes = len([n for n in self.strategy_handle.graph_of_alternatives.get_absorbing_state()])
        set_of_leaf_nodes = set([n for n in self.strategy_handle.graph_of_alternatives.get_absorbing_state()])
        self.assertEqual(num_of_nodes, LTLF_EXPECTED_GoAlt_OP[self.formula]["num_of_nodes"], msg=f"Mismatch in number of nodes in Graph of Alternatives. Should be {LTLF_EXPECTED_GoAlt_OP[self.formula]['num_of_nodes']} but got {num_of_nodes}")
        self.assertEqual(num_of_edges, LTLF_EXPECTED_GoAlt_OP[self.formula]["num_of_edges"], msg=f"Mismatch in number of edges in Graph of Alternatives. Should be {LTLF_EXPECTED_GoAlt_OP[self.formula]['num_of_edges']} but got {num_of_edges}")
        self.assertEqual(num_of_leaf_nodes, LTLF_EXPECTED_GoAlt_OP[self.formula]["num_of_leaf_nodes"], msg=f"Mismatch in number of leaf nodes in Graph of Alternatives. Should be {LTLF_EXPECTED_GoAlt_OP[self.formula]['num_of_leaf_nodes']} but got {num_of_leaf_nodes}")
        self.assertEqual(set_of_leaf_nodes, LTLF_EXPECTED_GoAlt_OP[self.formula]["leaf_nodes"], msg=f"Mismatch in set of leaf nodes in Graph of Alternatives. Should be {LTLF_EXPECTED_GoAlt_OP[self.formula]['leaf_nodes']} but got {set_of_leaf_nodes}")

    def test_09_reg_str_synthesis(self):
        """
         Tests Regret-minimizing strategy synthesis by playing Min-Max game on the Graph of Best Response. 

         Note: This method only check the correctness of the code. The validity of the synthesized strategy is done by rolling out the strategy.
        """
        print("Testing Regret-minimizing strategy synthesis by playing Min-Max game on the Graph of Best Response.")
        reg_value: int = self.strategy_handle.state_values[self.strategy_handle.graph_of_alternatives.get_initial_states()[0][0]]
        self.assertEqual(self.strategy_handle.graph_of_alternatives_fp_iter, LTLF_EXPECTED_REG_STR_OP[self.formula]["iters"], msg=f"Mismatch in # of iterations for Regret Computation. Should be {LTLF_EXPECTED_REG_STR_OP[self.formula]['iters']} but got {self.strategy_handle.graph_of_alternatives_fp_iter}")
        self.assertEqual(reg_value, LTLF_EXPECTED_REG_STR_OP[self.formula]["rval"], msg=f"Mismatch in Regret value of the game. Should be {LTLF_EXPECTED_REG_STR_OP[self.formula]['rval']} but got {reg_value}")
    
    def test_10_reg_str_validity(self):
        """
        Test if the optimal behaviors are in-line with the expected behaviors.
        """
        pass


class TestUnrealizableRegretStrSynth(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        graph_handle = ConstrcutGraphs(formula="F(p18)")
        cls.formula = "F(p18)"
        cls.causal_graph_instance = graph_handle.causal_graph_instance
        cls.transition_system_instance = graph_handle.transition_system_instance
        cls.two_player_instance = graph_handle.two_player_instance
        cls.dfa_handle = graph_handle.dfa_handle
        cls.product_graph = graph_handle.product_graph
    

    def test_0_unrealizable_task(self):
        """
         Test unrealizable task. Raises Overflowerror as math.ceil() fails in
           RegretMinimizationStrategySynthesis.edge_weighted_arena_finite_reg_solver()
        """
        with self.assertRaises(OverflowError):
            run_synthesis_and_rollout(strategy_type="Regret",
                                      game=self.product_graph,
                                      rollout_flag=False)



def create_suite():
    suite = unittest.TestSuite()
    for formula in LTLF_EXPECTED_DFA_OP.keys():
        # method that does the precomputation
        graph_handle = ConstrcutGraphs(formula=formula)
        strategy_handle, _ = run_synthesis_and_rollout(strategy_type="Regret",
                                                       game=graph_handle.product_graph,
                                                       rollout_flag=False)
        
        # Added the formula and the graph handles as class attributes
        TestRegretStrSynth.formula = formula
        TestRegretStrSynth.causal_graph_instance = graph_handle.causal_graph_instance
        TestRegretStrSynth.transition_system_instance = graph_handle.transition_system_instance
        TestRegretStrSynth.two_player_instance = graph_handle.two_player_instance
        TestRegretStrSynth.dfa_handle = graph_handle.dfa_handle
        TestRegretStrSynth.product_graph = graph_handle.product_graph
        TestRegretStrSynth.strategy_handle = strategy_handle

        # create the Test suite
        suite.addTest(unittest.makeSuite(TestRegretStrSynth))
    
    suite.addTest(unittest.makeSuite(TestUnrealizableRegretStrSynth))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(create_suite())