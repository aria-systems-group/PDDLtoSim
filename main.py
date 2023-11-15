import os
import time
import random
import datetime
import tracemalloc
import yaml
import warnings

from typing import Tuple, Optional, List, Dict

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.two_player_game import TwoPlayerGame

# call the regret synthesis code
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import\
    RegretMinimizationStrategySynthesis as RegMinStrSyn
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_safe_reach import QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort

from src.rollout_provider import RegretStrategyRolloutProvider
from src.execute_str import execute_saved_str, execute_str

from utls import timer_decorator

# define a constant to dump the yaml file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def compute_adv_strs(product_graph: TwoPlayerGraph,
                     purely_adv: bool = False,
                     random_human: bool = True,
                     no_intervention: bool = False,
                     cooperative: bool = False,
                     print_sim_str: bool = True) -> List:
    """
    A method to play the adversarial game.
    """
    start = time.time()
    # comp_mcr_solver = ValueIteration(product_graph, competitive=True)
    # comp_mcr_solver.solve(debug=True, plot=False)
    # assert comp_mcr_solver.is_winning() is True, "[Error] There does not exist a winning strategy!"

    # OG Algorithms
    # be_handle = QualitativeBestEffortReachSyn(game=product_graph, debug=True)
    # be_handle = QuantitativeBestEffortReachSyn(game=product_graph, debug=True)

    # My propsoed algorithm 
    # be_handle = QualitativeSafeReachBestEffort(game=product_graph, debug=True)
    be_handle = QuantitativeSafeReachBestEffort(game=product_graph, debug=True)

    be_handle.compute_best_effort_strategies(plot=False)
    if be_handle.is_winning() is not True: print("[Warning] There does not exist a winning strategy!")
    stop = time.time()
    print(f"******************************Min-Max Computation time: {stop - start} ****************************")

    # coop_val_dict = coop_mcr_solver.state_value_dict
    # comp_str_dict = comp_mcr_solver.str_dict
    comp_str_dict = be_handle.sys_best_effort_str

    _init_state = product_graph.get_initial_states()[0][0]

    _next_state = random.choice(list(comp_str_dict[_init_state])) if isinstance(comp_str_dict[_init_state], set) else comp_str_dict[_init_state]
    # print(f"Edge weight: {product_graph.get_edge_weight(_init_state, _next_state)}")
    _action_seq = []

    _action_seq.append(product_graph._graph[_init_state][_next_state][0].get("actions"))
    # print(_action_seq[-1])

    if purely_adv:
        while _next_state is not None:
            _curr_state = _next_state

            _next_state = random.choice(comp_str_dict.get(_curr_state))  if isinstance(comp_str_dict[_curr_state], List) else comp_str_dict[_curr_state]

            # print(f"Edge weight: {product_graph.get_edge_weight(_curr_state, _next_state)}")

            if _next_state is not None:
                _edge_act = product_graph._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(product_graph._graph[_curr_state][_next_state][0].get("actions"))
                    # print(_action_seq[-1])
    
    elif random_human:
         while _next_state is not None:
            _curr_state = _next_state

            if isinstance(_curr_state, tuple):
                if 'accept' in _curr_state[1]:
                    break

                if 'T0_S4' in _curr_state[1]:
                    break
            else:    
                if 'accept' in _curr_state:
                    break

                if 'T0_S4' in _curr_state:
                    break

            if product_graph.get_state_w_attribute(_curr_state, 'player') == "adam":
                _succ_states: List[tuple] = [_state for _state in product_graph._graph.successors(_curr_state)]
                _next_state = random.choice(_succ_states)
                c = 0 
                while ('accept' in _next_state or 'T0_S4' in _next_state) and c <= 5:
                    _next_state = random.choice(_succ_states)
                    c += 1
            
            else:
                _next_state = random.choice(list(comp_str_dict.get(_curr_state)))  if isinstance(comp_str_dict[_curr_state], set) else comp_str_dict[_curr_state]

            # print(f"Edge weight: {product_graph.get_edge_weight(_curr_state, _next_state)}")

            if _next_state is not None:
                _edge_act = product_graph._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(product_graph._graph[_curr_state][_next_state][0].get("actions"))

    elif no_intervention:
        while _next_state is not None:
            _curr_state = _next_state

            if product_graph.get_state_w_attribute(_curr_state, "player") == "adam":
                # get the state that sys wanted to evolve to
                for _succ in product_graph._graph.successors(_curr_state):
                    _edge_action = product_graph._graph[_curr_state][_succ][0]["actions"]
                    _edge_type = get_action_from_causal_graph_edge(_edge_action)
                    if _edge_type != "human-move":
                        _next_state = _succ
                        break
            else:
                _next_state = random.choice(comp_str_dict.get(_curr_state))

            if _next_state is not None:
                _edge_action = product_graph._graph[_curr_state][_next_state][0].get('actions')
                if _action_seq[-1] != _edge_action:
                    _action_seq.append(_edge_action)

    elif cooperative:
        _coop_str_dict = compute_cooperative_actions_for_env(product_graph)
        _max_coop_actions: int = 0
        while _next_state is not None:
            _curr_state = _next_state

            if product_graph.get_state_w_attribute(_curr_state, attribute="player") == "eve":
                _next_state = random.choice(comp_str_dict.get(_curr_state))
            else:
                if _max_coop_actions <= 2:
                    _next_state = _coop_str_dict[_curr_state]
                    # only increase the counter when the human moves
                    _max_coop_actions += 1
                else:
                    for _succ in product_graph._graph.successors(_curr_state):
                        _edge_action = product_graph._graph[_curr_state][_succ][0]["actions"]
                        _edge_type = get_action_from_causal_graph_edge(_edge_action)
                        if _edge_type != "human-move":
                            _next_state = _succ
                            break

            if _next_state is not None:
                _edge_act = product_graph._graph[_curr_state][_next_state][0].get("actions")
                if _action_seq[-1] != _edge_act:
                    _action_seq.append(product_graph._graph[_curr_state][_next_state][0].get("actions"))

    else:
        warnings.warn("Please at-least one of the flags i.e Cooperative, no_intervention or purely_adversarial is True")

    if print_sim_str:
        for _action in _action_seq:
            print(_action)

    return _action_seq


# def compute_reg_strs(product_graph: TwoPlayerGraph,
#                      coop_str: bool = False,
#                      epsilon: float = -1) -> Tuple[list, dict, TwoPlayerGraph]:
#     """
#     A method to compute strategies. We control the env's behavior by making it purely cooperative, pure adversarial, or
#     epsilon greedy.

#     @param coop_str: Set this to be true for purely cooperative behavior from the env
#     @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks
#      random actions.
#     """
#     # build an instance of regret strategy minimization class
#     reg_syn_handle = RegMinStrSyn(product_graph)
#     reg_str, reg_val = reg_syn_handle.edge_weighted_arena_finite_reg_solver(reg_factor=1.25,
#                                                                             purge_states=True,
#                                                                             plot=False)
#     twa_game = reg_syn_handle.graph_of_alternatives
#     _init_state = twa_game.get_initial_states()[0][0]
#     for _n in twa_game._graph.successors(_init_state):
#         print(f"Reg Val: {_n}: {reg_val[_n]}")
#     # the reg str is dict that one from one state to another. Lets convert this to print a sequence of edge actions
#     _next_state = reg_str[_init_state]
#     _action_seq = []

#     _action_seq.append(twa_game._graph[_init_state][_next_state][0].get("actions"))

#     if coop_str:
#         # compute cooperative strs for the player
#         _coop_str_dict = compute_cooperative_actions_for_env(twa_game)
#         _max_coop_actions: int = 1

#         # print(f"{_init_state}: {reg_val[_init_state]}")
#         # print(f"{_next_state}: {reg_val[_init_state]}")
#         while _next_state is not None:
#             _curr_state = _next_state

#             if twa_game.get_state_w_attribute(_curr_state, attribute="player") == "eve":
#                 _next_state = reg_str.get(_curr_state)
#             else:
#                 if _max_coop_actions <= 10:
#                     _next_state = _coop_str_dict[_curr_state]
#                     # only increase the counter when the human moves
#                     _max_coop_actions += 1
#                 else:
#                     _next_state = reg_str.get(_curr_state)

#             if _next_state is not None:
#                 _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
#                 if _action_seq[-1] != _edge_act:
#                     _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))
#                 # print(f"{_next_state}: {reg_val[_init_state]}")

#     elif 0 <= epsilon <= 1:
#         # we randomise human strategies
#         _new_str_dict = compute_epsilon_str_dict(epsilon=epsilon,
#                                                  reg_str_dict=reg_str,
#                                                  max_human_int=3, twa_game=twa_game)
#         while _next_state is not None:
#             _curr_state = _next_state

#             _next_state = _new_str_dict.get(_curr_state)

#             if _next_state is not None:
#                 _edge_act = twa_game._graph[_curr_state][_next_state][0].get("actions")
#                 if _action_seq[-1] != _edge_act:
#                     _action_seq.append(twa_game._graph[_curr_state][_next_state][0].get("actions"))

#     for _action in _action_seq:
#         print(_action)

#     return _action_seq, reg_val, twa_game



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
    _project_root = os.path.dirname(os.path.abspath(__file__))

    _domain_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    # _problem_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/problem.pddl"
    _problem_file_path = _project_root + "/pddl_files/two_table_scenario/diagonal/sym_test_problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")
    start = time.time()
    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    # _transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.edges())}")

    _two_player_instance = TwoPlayerGame(_causal_graph_instance, _transition_system_instance)
    _two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=False)

    # product_graph = two_player_instance.build_product(dfa=dfa, trans_sys=two_player_instance.two_player_game)

    # for implicit construction, the human intervention should >=2
    _two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    _two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    _two_player_instance.modify_ap_w_object_types(implicit=True)
    _two_player_instance.modify_edge_weights(implicit=True)
    stop = time.time()
    print(f"******************************Original Graph construction time: {stop - start}******************************")

    # print # of Sys and Env state
    env_count = 0
    sys_count = 0
    for (p, d) in _two_player_instance._two_player_implicit_game._graph.nodes(data=True):
        if d['player'] == 'adam':
            env_count += 1
        elif d['player'] == 'eve':
            sys_count += 1

    print(f"# of Sys states in Two player game: {sys_count}")
    print(f"# of Env states in Two player game: {env_count}")

    # for (u, v) in _two_player_instance._two_player_implicit_game._graph.edges():
    #     print(f"{u} -------{_two_player_instance._two_player_implicit_game._graph[u][v][0].get('actions')}------> {v}")
    # sys.exit(-1)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.edges())}")

    # _dfa = _two_player_instance.build_LTL_automaton(formula="F(l2 || l6)")
    # _dfa = _two_player_instance.build_LTL_automaton(
    #     formula="F((p22 & p14 & p03) || (p05 & p19 & p26))")
    _dfa = _two_player_instance.build_LTL_automaton(
        # formula="F((p03 & p14 & p22) || (p05 & p19 & p26))")
        # formula="F(p01) & F(p17) & G(!p18)")
        formula="F(p01 || p17)")
    # _dfa = _two_player_instance.build_LTL_automaton(
    #     formula="F((p12 & p00) || (p20 & p12) || (p05 & p19) || (p25 & p19))")

    _product_graph = _two_player_instance.build_product(dfa=_dfa,
                                                        trans_sys=_two_player_instance.two_player_implicit_game)
    _relabelled_graph = _two_player_instance.internal_node_mapping(_product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(_relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(_relabelled_graph._graph.edges())}")
    

    # compute strs
    # _actions, _reg_val, _graph_of_alts = compute_reg_strs(_product_graph, coop_str=False, epsilon=0)
    # _manual_rollout(product_graph=_product_graph)

    # build an instance of regret strategy minimization class
    reg_syn_handle = RegMinStrSyn(_product_graph)
    strategy, state_vals = reg_syn_handle.edge_weighted_arena_finite_reg_solver(reg_factor=1.25,
                                                                                purge_states=True,
                                                                                plot=False)
    rollout_handle = RegretStrategyRolloutProvider(twa_game=reg_syn_handle.graph_of_alternatives,
                                                   dfa_game=_product_graph,
                                                   regret_strategy=strategy,
                                                   regret_sv=state_vals,
                                                   debug=True)
    
    # testing all rollout provider
    # rollout_handle.get_manual_rollout()
    
    # rollout_handle.get_rollout_with_human_intervention()
    rollout_handle.get_rollout_with_epsilon_human_intervention(epsilon=1)
    # rollout_handle.get_rollout_with_epsilon_human_intervention(epsilon=0.5)

    return

    # adversarial strs
    # _actions = compute_adv_strs(_product_graph,
    #                             purely_adv=False,
    #                             random_human=True,
    #                             no_intervention=False,
    #                             cooperative=False,
    #                             print_sim_str=True)
    
    

    # ask the user if they want to save the str or not
    # _dump_strs = input("Do you want to save the strategy,Enter: Y/y")
    _dump_strs = "n"
    # save strs
    # if _dump_strs == "y" or _dump_strs == "Y":
    #     save_str(causal_graph=_causal_graph_instance,
    #              transition_system=_transition_system_instance,
    #              two_player_game=_two_player_instance,
    #              regret_graph_of_alternatives=_graph_of_alts,
    #              game_reg_value=_reg_val,
    #              pos_seq=_actions,
    #              adversarial=False)

    # simulate the str
    # execute_str(actions=_actions,
    #             causal_graph=_causal_graph_instance,
    #             transition_system=_transition_system_instance,
    #             exp_name="diag",
    #             record_sim=record_flag,
    #             debug=False)


def arch_main(print_flag: bool = False, record_flag: bool = False) -> None:
    _project_root = os.path.dirname(os.path.abspath(__file__))

    _domain_file_path = _project_root + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = _project_root + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")

    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    _transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)
    _transition_system_instance.modify_edge_weights()

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.edges())}")

    _two_player_instance = TwoPlayerGame(_causal_graph_instance, _transition_system_instance)
    _two_player_instance.build_two_player_game(human_intervention=2,
                                               human_intervention_cost=0,
                                               plot_two_player_game=False,
                                               arch_construction=True)

    # for implicit construction, the human intervention should >=2
    _two_player_instance.build_two_player_implicit_transition_system_from_explicit(
        plot_two_player_implicit_game=False)
    _two_player_instance.set_appropriate_ap_attribute_name(implicit=True)
    # two_player_instance.modify_ap_w_object_types(implicit=True)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(_two_player_instance._two_player_implicit_game._graph.edges())}")

    _dfa = _two_player_instance.build_LTL_automaton(formula="F((l8 & l9 & l0) || (l3 & l2 & l1))")
    # _dfa = _two_player_instance.build_LTL_automaton(formula="F(l8 & l9 & l0)")
    # _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                                     trans_sys=_two_player_instance.two_player_game)

    _product_graph = _two_player_instance.build_product(dfa=_dfa,
                                                        trans_sys=_two_player_instance.two_player_implicit_game)

    _relabelled_graph = _two_player_instance.internal_node_mapping(_product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(_relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(_relabelled_graph._graph.edges())}")

    # compute strs
    # _actions, _reg_val, _graph_of_alts = compute_reg_strs(_product_graph, coop_str=False, epsilon=0)

    # adversarial strs
    # _actions = compute_adv_strs(_product_graph,
    #                             purely_avd=True,
    #                             no_intervention=False,
    #                             cooperative=False,
    #                             print_sim_str=True)
    exit()

    # ask the user if they want to save the str or not
    _dump_strs = input("Do you want to save the strategy,Enter: Y/y")

    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=_causal_graph_instance,
                 transition_system=_transition_system_instance,
                 two_player_game=_two_player_instance,
                 # regret_graph_of_alternatives=_graph_of_alts,
                 # game_reg_value=_reg_val,
                 pos_seq=_actions,
                 adversarial=True)

    # simulate the str
    execute_str(actions=_actions,
                causal_graph=_causal_graph_instance,
                transition_system=_transition_system_instance,
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