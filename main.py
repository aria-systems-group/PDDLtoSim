import os

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.abspath(__file__))

    # Experimental stage - lets try calling the function within pyperplan
    domain_file_path = _project_root + "/pddl_files/blocks_world/domain.pddl"
    problem_file_ath = _project_root + "/pddl_files/blocks_world/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=problem_file_ath, domain_file=domain_file_path, draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False)

    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False)

    _two_player_instance = TwoPlayerGame(_causal_graph_instance, _transition_system_instance)
    _two_player_instance.build_two_player_game(plot_two_player_game=True)