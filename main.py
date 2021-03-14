import os
import warnings
import pybullet as pb
import numpy as np

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

from src.pddl_env_simualtor.envs.panda_sim import PandaSim

def load_pre_build_loc_info(exp_name: str):
    if exp_name == "diag":
        loc_dict = {
            'l0': np.array([0.3, 0.2, 0.17/2]),
            'l1': np.array([0.3, -0.2, 0.17/2]),
            'l2': np.array([0.6, 0.2, 0.17/2]),
            'l3': np.array([0.6, -0.2, 0.17/2]),
            'l4': np.array([0.45, 0.0, 0.17/2])
        }
    elif exp_name == "arch":
        pass
    else:
        warnings.warn("PLease enter a valid experiment name")

if __name__ == "__main__":
    # build the product automaton
    _project_root = os.path.dirname(os.path.abspath(__file__))

    # Experimental stage - lets try calling the function within pyperplan
    domain_file_path = _project_root + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/pddl_files/blocks_world/problem.pddl"

    causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=False)

    causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)

    transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    transition_system_instance.build_transition_system(plot=False)

    two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    two_player_instance.build_two_player_game(human_intervention=5, plot_two_player_game=False)
    two_player_instance.set_appropriate_ap_attribute_name()
    two_player_instance.modify_ap_w_object_types()

    dfa = two_player_instance.build_LTL_automaton(formula="F(p01 & ((p11 & p22) || (p12 & p21)))")
    product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)
    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)
    # relabelled_graph.plot_graph()

    # print some details about the product graph
    print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
    print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    # build the simulator
    physics_client = pb.connect(pb.GUI)

    panda = PandaSim(physics_client, use_IK=1)