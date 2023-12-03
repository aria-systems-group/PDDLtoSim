import os
import sys
import time
import datetime
import tracemalloc
import yaml
import warnings

from typing import Optional, Dict, Type

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

# call the regret synthesis code
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
from regret_synthesis_toolbox.src.graph.product import ProductAutomaton
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import\
    RegretMinimizationStrategySynthesis as RegMinStrSyn
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_safe_reach import QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort

from src.rollout_provider import rollout_strategy, RolloutProvider
from src.execute_str import execute_saved_str, execute_str

from config import *
from utls import timer_decorator

# define a constant to dump the yaml file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def compute_strategy(strategy_type: str, game: ProductAutomaton, debug: bool = False, plot: bool = False, reg_factor: float = 1.25):
    """
     A method that call the appropriate strategy synthesis class nased on the user input. 

     Valid strategy_type: Min-Max, Min-Min, Regret, BestEffortQual, BestEffortQuant, BestEffortSafeReachQual, BestEffortSafeReachQuant
    """
    valid_str_syn_algos = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "BestEffortSafeReachQual", "BestEffortSafeReachQuant"]

    if strategy_type == "Min-Max":
        strategy_handle = ValueIteration(game, competitive=True)
        strategy_handle.solve(debug=debug, plot=plot)

    elif strategy_type == "Min-Min":
        strategy_handle = ValueIteration(game, competitive=False)
        strategy_handle.solve(debug=debug, plot=plot)
    
    elif strategy_type == "Regret":
        strategy_handle = RegMinStrSyn(game)
        strategy_handle.edge_weighted_arena_finite_reg_solver(reg_factor=reg_factor,
                                                              purge_states=True,
                                                              plot=plot)
    
    elif strategy_type == "BestEffortQual":
        strategy_handle = QualitativeBestEffortReachSyn(game=game, debug=debug)    
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    # My propsoed algorithms
    elif strategy_type == "BestEffortQuant":
        strategy_handle = QuantitativeBestEffortReachSyn(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    elif strategy_type == "BestEffortSafeReachQual":
        strategy_handle = QualitativeSafeReachBestEffort(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    elif strategy_type == "BestEffortSafeReachQuant":
        strategy_handle = QuantitativeSafeReachBestEffort(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)

    else:
        warnings.warn(f"[Error] Please enter a valid Strategy Synthesis variant:[ {', '.join(valid_str_syn_algos)} ]")
        sys.exit(-1)

    return strategy_handle


def save_str(causal_graph: CausalGraph,
             transition_system: FiniteTransitionSystem,
             two_player_game: TwoPlayerGame,
             pos_seq: list,
             regret_graph_of_alternatives: Optional[TwoPlayerGraph] = None,
             game_reg_value: Optional[dict] = None,
             adversarial: bool = False):
    """
    A helper method that dumps the regret value and the corresponding strategy computed for given abstraction and an
    LTL formula. This method creates a yaml file which is then dumped in the saved_strs folder at the root of the
    project. The file naming convention is
    <task_name>_<# of boxes>_<# of locs>_<# of possible human intervention>_<reg_value>.

    The stuff being dumped is :
        1. Task Name
        2. # of boxes along with the names
        3. # of locs along with the names
        4. # of possible human interventions
        5. # of nodes in the game
        6. # of edges in the game
        7. LTL Formula used
        8. strategy compute (a sequence of actions)
    """
    
    _task_name: str = causal_graph.get_task_name()
    _boxes = causal_graph.task_objects
    _locations = causal_graph.task_locations

    if not adversarial:
        _init_state = regret_graph_of_alternatives.get_initial_states()[0][0]
        _reg_value: Optional[int] = game_reg_value.get(_init_state)
    else:
        _reg_value = None

    _init_state = transition_system.transition_system.get_initial_states()[0][0]
    _init_conf = transition_system.transition_system.get_state_w_attribute(_init_state, "list_ap")

    _possible_human_interventions: int = two_player_game.human_interventions

    # transition system nodes and edges
    _trans_sys_nodes = len(transition_system.transition_system._graph.nodes())
    _trans_sys_edges = len(transition_system.transition_system._graph.edges())

    # product graph nodes and edges
    _prod_nodes = len(two_player_game.two_player_game._graph.nodes())
    _prod_edges = len(two_player_game.two_player_game._graph.edges())

    if not adversarial:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = len(regret_graph_of_alternatives._graph.nodes())
        _graph_of_alts_edges = len(regret_graph_of_alternatives._graph.edges())
    else:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = None
        _graph_of_alts_edges = None

    _ltl_formula = two_player_game.formula

    # create a data dict to dump it
    data_dict: Dict = dict(
        task_name=_task_name,
        no_of_boxes={
            'num': len(_boxes),
            'objects': _boxes,
        },
        no_of_loc={
            'num': len(_locations),
            'objects': _locations
        },
        max_human_int=_possible_human_interventions,
        init_worl_conf=_init_conf,
        ltl_formula=_ltl_formula,
        abstractions={
            'num_transition_system_nodes': _trans_sys_nodes,
            'num_transition_system_edges': _trans_sys_edges,
            'num_two_player_game_nodes': _prod_nodes,
            'num_two_player_game_edges': _prod_edges,
            'num_graph_of_alts_nodes': _graph_of_alts_nodes,
            'num_graph_of_alts_edges': _graph_of_alts_edges,
        },
        reg_val=_reg_value,
        reg_str=pos_seq
    )

    # now dump the data in a file
    if adversarial:
        _file_name: str = \
            f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_{_possible_human_interventions}_h_" \
            f"{_reg_value}_adv_"
    else:
        _file_name: str =\
        f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_{_possible_human_interventions}_h_" \
        f"{_reg_value}_reg_"

    _current_date_time_stamp = str(datetime.datetime.now())

    # remove the seconds stamp
    _time_stamp, *_ = _current_date_time_stamp.partition('.')
    _time_stamp = _time_stamp.replace(" ", "_" )
    _time_stamp = _time_stamp.replace(":", "_")
    _time_stamp = _time_stamp.replace("-", "_")
    _file_path = ROOT_PATH + _file_name + _time_stamp + ".yaml"
    try:
        with open(_file_path, 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False, sort_keys=False)

    except FileNotFoundError:
        print(FileNotFoundError)
        print(f"The file {_file_path} could not be found."
              f" This could be because I could not find the folder to dump in")


def load_data_from_yaml_file(file_add: str) -> Dict:
    """
    A helper function to load the sequence of strategies given a valid yaml file.
    """

    try:
        with open(file_add, 'r') as stream:
            graph_data = yaml.load(stream, Loader=yaml.Loader)

    except FileNotFoundError as error:
        print(error)
        print(f"The file does not exist at the loc {file_add}")

    return graph_data


@timer_decorator
def daig_main(print_flag: bool = False, record_flag: bool = False) -> None:
    domain_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    # _problem_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/problem.pddl"
    problem_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/sym_test_problem.pddl"

    causal_graph_instance = CausalGraph(problem_file=problem_file_path,
                                         domain_file=domain_file_path,
                                         draw=False)

    causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.edges())}")
    start = time.time()
    transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    # transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(transition_system_instance.transition_system._graph.edges())}")

    two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=False)

    # product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)

    # for implicit construction, the human intervention should >=2
    two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    two_player_instance.modify_ap_w_object_types(implicit=True)
    # two_player_instance.modify_edge_weights(implicit=True)
    stop = time.time()
    print(f"******************************Original Graph construction time: {stop - start}******************************")

    # print # of Sys and Env state
    env_count = 0
    sys_count = 0
    for (p, d) in two_player_instance._two_player_implicit_game._graph.nodes(data=True):
        if d['player'] == 'adam':
            env_count += 1
        elif d['player'] == 'eve':
            sys_count += 1

    print(f"# of Sys states in Two player game: {sys_count}")
    print(f"# of Env states in Two player game: {env_count}")

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.edges())}")

    dfa = two_player_instance.build_LTL_automaton(formula=FORMULA_1B_3L_AND_W_TRAP)

    product_graph = two_player_instance.build_product(dfa=dfa,
                                                      trans_sys=two_player_instance.two_player_implicit_game)
    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)

    # edge_weights = set({})
    # for (u, v, d) in product_graph._graph.edges(data=True):
    #     if d['weight'] == 0 and 'human' not in d['actions'] and not product_graph._graph.nodes(data=True)[u]['player'] == 'adam':
    #         print(f"Action {d['actions']}")
    #     edge_weights.add(d['weight'])
    
    # print(f"Edge weights in the product graph: {edge_weights}")
    # exit()

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")
    
    # create a strategy synthesis handle and solve the game
    # valid_str_syn_algos = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "BestEffortSafeReachQual", "BestEffortSafeReachQuant"]
    # valid_human_stings = ["no-human", "random-human", "epsilon-human"]
    # for st in valid_str_syn_algos:
    #     for hs in valid_human_stings:
    #         strategy_handle = compute_strategy(strategy_type=st, game=product_graph, debug=False, plot=False)

    #         # rollout the stratgey
    #         roller: Type[RolloutProvider] = rollout_strategy(strategy=strategy_handle,
    #                                                          game=product_graph,
    #                                                          debug=False,
    #                                                          human_type=hs)
    
    strategy_handle = compute_strategy(strategy_type="BestEffortSafeReachQuant", game=product_graph, debug=False, plot=False)

    # rollout the stratgey
    roller: Type[RolloutProvider] = rollout_strategy(strategy=strategy_handle,
                                                     game=product_graph,
                                                     debug=True,
                                                     human_type="no-human")

    # return
    # ask the user if they want to save the str or not
    _dump_strs = input("Do you want to save the strategy,Enter: Y/y")
    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=causal_graph_instance,
                 transition_system=transition_system_instance,
                 two_player_game=two_player_instance,
                 pos_seq=roller.action_seq,
                 adversarial=False)

    # # simulate the str
    # execute_str(actions=roller.action_seq,
    #             causal_graph=causal_graph_instance,
    #             transition_system=transition_system_instance,
    #             exp_name="diag",
    #             record_sim=record_flag,
    #             debug=False)


@timer_decorator
def arch_main(print_flag: bool = False, record_flag: bool = False) -> None:
    domain_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/arch/domain.pddl"
    problem_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/arch/problem.pddl"

    causal_graph_instance = CausalGraph(problem_file=problem_file_path,
                                         domain_file=domain_file_path,
                                         draw=False)

    causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(causal_graph_instance._causal_graph._graph.edges())}")

    transition_system_instance = FiniteTransitionSystem(causal_graph_instance)
    transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)
    transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(transition_system_instance.transition_system._graph.edges())}")

    two_player_instance = TwoPlayerGame(causal_graph_instance, transition_system_instance)
    two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=True)

    # for implicit construction, the human intervention should >=2
    two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    # two_player_instance.modify_ap_w_object_types(implicit=True)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.edges())}")

    dfa = two_player_instance.build_LTL_automaton(formula="F((l8 & l9 & l0) || (l3 & l2 & l1))")
    # product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                                     trans_sys=_two_player_instance.two_player_game)

    product_graph = two_player_instance.build_product(dfa=dfa,
                                                      trans_sys=two_player_instance.two_player_implicit_game)

    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    # exit()
    # create a strategy synthesis handle and solve the game
    valid_str_syn_algos = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "BestEffortSafeReachQual", "BestEffortSafeReachQuant"]
    valid_human_stings = ["no-human", "random-human", "epsilon-human"]
    for st in valid_str_syn_algos:
        for hs in valid_human_stings:
            strategy_handle = compute_strategy(strategy_type=st, game=product_graph, debug=False, plot=False)

            # rollout the stratgey
            roller: Type[RolloutProvider] = rollout_strategy(strategy=strategy_handle,
                                                             game=product_graph,
                                                             debug=False,
                                                             human_type=hs)

    # ask the user if they want to save the str or not
    _dump_strs = input("Do you want to save the strategy,Enter: Y/y")

    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=causal_graph_instance,
                 transition_system=transition_system_instance,
                 two_player_game=two_player_instance,
                 pos_seq=roller.action_seq,
                 adversarial=True)

    # simulate the str
    execute_str(actions=roller.action_seq,
                causal_graph=causal_graph_instance,
                transition_system=transition_system_instance,
                exp_name="arch",
                record_sim=record_flag,
                debug=False)


if __name__ == "__main__":
    record = False
    use_saved_str = False

    if use_saved_str:
        # get the actions from the yaml file
        file_name = "/arch_2_tables_4_box_6_loc_2_h_2_reg_2021_04_28_14_23_52.yaml"
        file_pth: str = ROOT_PATH + "/saved_strs" + file_name

        yaml_dump = load_data_from_yaml_file(file_add=file_pth)

        execute_saved_str(yaml_data=yaml_dump,
                          exp_name="arch",
                          record_sim=record,
                          debug=False)

    else:
        # starting the monitoring
        tracemalloc.start()
        daig_main(print_flag=True, record_flag=record)
        # arch_main(print_flag=False, record_flag=record)

        # displaying the memory - output current memory usage and peak memory usage
        _,  peak_mem = tracemalloc.get_traced_memory()
        print(f" Peak memory [MB]: {peak_mem/(1024*1024)}")
        # stopping the library
        tracemalloc.stop()