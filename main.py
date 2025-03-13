import os
import sys
import copy
import time
import argparse
import datetime
import tracemalloc
import yaml

import warnings

from collections import OrderedDict
from typing import Optional, Dict, Type, Union, Tuple, List

from icra_examples.safe_adm_game import modify_abstraction, remove_non_reachable_states
from icra_examples.tic_tac_toe_abs import TicTacToe

from src.graph_construction.causal_graph import CausalGraph
from src.graph_construction.two_player_game import TwoPlayerGame
from src.graph_construction.transition_system import FiniteTransitionSystem
from src.graph_construction.minigrid_two_player_game import NonDeterministicMiniGrid

# for logging purposes
from regret_synthesis_toolbox.src.simulation.simulator import Simulator

# call the regret synthesis code
from regret_synthesis_toolbox.helper import InteractiveGraph
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph, Graph
from regret_synthesis_toolbox.src.graph.product import ProductAutomaton
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import\
    RegretMinimizationStrategySynthesis as RegMinStrSyn
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantitativeNaiveAdmissible, QuantitativeGoUAdmissible, QuantitativeGoUAdmissibleWinning
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantiativeRefinedAdmissible, QuantitativeAdmMemorless

from src.execute_str import execute_saved_str
from src.rollout_str.rollout_provider_if import RolloutProvider
from src.rollout_str.rollout_main import rollout_strategy, VALID_ENV_STRINGS, Strategy

from config import *
from utls import timer_decorator

# define a constant to dump the yaml file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DfaGame = Union[TwoPlayerGraph, TwoPlayerGame, NonDeterministicMiniGrid]

VALID_STR_SYN_ALGOS = ["Min-Max", "Min-Min", "Regret", "BestEffortQual", "BestEffortQuant", "QuantitativeNaiveAdmissible", \
                        "QuantitativeGoUAdmissible", "QuantitativeGoUAdmissibleWinning", "QuantiativeRefinedAdmissible", "QuantitativeAdmMemorless"]
VALID_ABSTRACTION_INSTANCES = ['daig-main', 'arch-main', 'minigrid', 'tic-tac-toe']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run minigrid_main with different parameters.')
    parser.add_argument('--human_type', type=str, required=True, help='The type of human: epsilon-human, random-human, coop-human, mixed-human')
    parser.add_argument('--strategy_type', type=str, required=True, help=f'The type of strategy: {VALID_STR_SYN_ALGOS[-2:]}')
    parser.add_argument('--env_type', type=str, required=False, default='', help='The type of Minigrid Env: MiniGrid-LavaAdm_karan-v0, MiniGrid-IntruderRobotRAL25-v0, MiniGrid-ThreeDoorIntruderRobotRAL25-v0, MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0')
    return parser.parse_args()

args = parse_arguments()


@timer_decorator
def compute_strategy(strategy_type: str, game: ProductAutomaton, debug: bool = False, plot: bool = False, reg_factor: float = 1.25) -> Strategy:
    """
     A method that call the appropriate strategy synthesis class nased on the user input. 

     Valid strategy_type: Min-Max, Min-Min, Regret, BestEffortQual, BestEffortQuant, BestEffortSafeReachQual, BestEffortSafeReachQuant

     TODO: Add support for Adversarial strategy synthesis and Cooperative strategy synthesis (both qualitative).
    """
    if strategy_type == "Min-Max":
        strategy_handle = ValueIteration(game, competitive=True)
        strategy_handle.solve(debug=debug, plot=plot)

    elif strategy_type == "Min-Min":
        strategy_handle = ValueIteration(game, competitive=False)
        strategy_handle.solve(debug=debug, plot=plot)
    
    elif strategy_type == "Regret":
        strategy_handle = RegMinStrSyn(game, reg_factor=reg_factor)
        strategy_handle.sanity_checking = True
        strategy_handle.edge_weighted_arena_finite_reg_solver(purge_states=True,
                                                              plot=plot)
    
    elif strategy_type == "BestEffortQual":
        strategy_handle = QualitativeBestEffortReachSyn(game=game, debug=debug)    
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    # My proposed algorithms
    elif strategy_type == "BestEffortQuant":
        strategy_handle = QuantitativeBestEffortReachSyn(game=game, debug=debug)
        strategy_handle.compute_best_effort_strategies(plot=plot)
    
    elif strategy_type == 'QuantitativeNaiveAdmissible':
        strategy_handle = QuantitativeNaiveAdmissible(budget=4, game=game, debug=debug)
        strategy_handle.compute_adm_strategies(plot=plot)
    
    elif strategy_type == "QuantitativeGoUAdmissible":
        print("************************Playing QuantitativeGoUAdmissible************************")
        strategy_handle = QuantitativeGoUAdmissible(budget=12, game=game, debug=debug)
        strategy_handle.compute_adm_strategies(plot=plot, compute_str=False)
    
    elif strategy_type == "QuantitativeGoUAdmissibleWinning":
        print("************************Playing QuantitativeGoUAdmissibleWinning************************")
        strategy_handle = QuantitativeGoUAdmissibleWinning(budget=12, game=game, debug=debug)
        strategy_handle.compute_adm_strategies(plot=plot, compute_str=False)
    
    elif strategy_type == "QuantiativeRefinedAdmissible":
        strategy_handle = QuantiativeRefinedAdmissible(game=game, debug=debug)
        strategy_handle.compute_adm_strategies(plot=plot)
    
    elif strategy_type == "QuantitativeAdmMemorless":
        strategy_handle = QuantitativeAdmMemorless(game=game, debug=debug)
        strategy_handle.compute_adm_strategies(plot=plot)
    
    else:
        warnings.warn(f"[Error] Please enter a valid Strategy Synthesis variant:[ {', '.join(VALID_STR_SYN_ALGOS)} ]")
        sys.exit(-1)

    return strategy_handle



def run_all_synthesis_and_rollouts(game: DfaGame, debug: bool = False, reg_factor: float = 1.25) -> None:
    """
    A helper function that compute all type of strategies from the set of valid strategies for all possible env (human) behaviors from the set of valid behaviors. 
    """
    # remove 'manual-rollout' for automated testing
    _env_string: List[str] = copy.deepcopy(VALID_ENV_STRINGS)
    _env_string.remove("manual")

    # create a strategy synthesis handle and solve the game
    for st in VALID_STR_SYN_ALGOS:
        print(f"******************************************Rolling out: {st} strategy******************************************")
        strategy_handle = compute_strategy(strategy_type=st, game=game, debug=debug, plot=False, reg_factor=reg_factor)
        
        # rollout the stratgey
        for hs in _env_string:
            print(f"******************************************With: {hs} env******************************************")
            rollout_strategy(strategy=strategy_handle,
                             game=game,
                             debug=False,
                             human_type=hs)


@timer_decorator
def run_synthesis_and_rollout(strategy_type: str,
                              game: DfaGame,
                              human_type: str = 'no-human',
                              sys_type: str = '',
                              rollout_flag: bool = False,
                              debug: bool = False,
                              epsilon: float = 0.1,
                              reg_factor: float = 1.25, 
                              max_iterations: int = 100) -> Tuple[Strategy, Optional[RolloutProvider]]:
    """
    A helper function that compute all type of strategies from the set of valid strategies for all possible env (human) behaviors from the set of valid behaviors. 
    """
    assert strategy_type in VALID_STR_SYN_ALGOS, f"[Error] Please enter a valid Strategy Synthesis variant:[ {', '.join(VALID_STR_SYN_ALGOS)} ]"
    
    if strategy_type in ["QuantitativeNaiveAdmissible", "QuantitativeGoUAdmissible", "QuantitativeGoUAdmissibleWinning"]:
        assert human_type == "manual" , "Trying to rollout Adm strategies. Currently you can only manually rollout. Please set 'human_type'='manual'."
    
    # create a strategy synthesis handle and solve the game
    str_handle = compute_strategy(strategy_type=strategy_type,
                                  game=game,
                                  debug=False,
                                  plot=False,
                                  reg_factor=reg_factor)

    assert human_type in VALID_ENV_STRINGS, f"[Error] Please enter a valid human type from:[ {', '.join(VALID_ENV_STRINGS)} ]"

    simulator = Simulator(env=None, game=game)

    # rollout the stratgey
    if rollout_flag:
        for _ in range(NUM_OF_TRAILS):
            roller: Type[RolloutProvider] = rollout_strategy(strategy=str_handle,
                                                            game=game,
                                                            debug=False,
                                                            human_type=human_type,
                                                            logger=simulator,
                                                            sys_type=sys_type,
                                                            epsilon=epsilon,
                                                            max_iterations=max_iterations)
            simulator._episode += 1
        simulator.get_stats()

        #dump the data for bookkeeping
        now = datetime.datetime.now()
        timestamp: str = now.strftime("%Y%m%d_%H%M%S")
        filename = "/" + game._graph.name + "_" + "NOT_CPLX_" + strategy_type + "_" + human_type + "_" + sys_type + "_" + str(epsilon) + timestamp + ".yaml" 
        # filename = "/" + game._graph.name + "_" + strategy_type + "_" + human_type + "_" + sys_type + "_" + str(epsilon) + timestamp + ".yaml" 
        simulator.dump_results_to_yaml(ROOT_PATH + BENCHMARK_DIR_WAIT + filename)
        return str_handle, roller
    
    return str_handle, None


def save_str(two_player_game: ProductAutomaton,
             pos_seq: list,
             causal_graph: Optional[CausalGraph] = None,
             dfa_game: Optional[ProductAutomaton] = None,
             transition_system: Optional[FiniteTransitionSystem] = None,
             regret_graph_of_alternatives: Optional[TwoPlayerGraph] = None,
             game_reg_value: Optional[dict] = None,
             admissible: bool = False,
             adversarial: bool = False,
             minigrid: bool = True) -> None:
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
    if causal_graph is not None:
        _task_name: str = causal_graph.get_task_name()
        _boxes = causal_graph.task_objects
        _locations = causal_graph.task_locations
    else:
        _task_name: str = two_player_game.graph_name
        _boxes = None
        _locations = None

    if regret_graph_of_alternatives is not None:
        _init_state = regret_graph_of_alternatives.get_initial_states()[0][0]
        _reg_value: Optional[int] = game_reg_value.get(_init_state)
    else:
        _reg_value = None
        _init_state = two_player_game.trans_sys.get_initial_states()[0][0]

    try:
        _init_conf = two_player_game.trans_sys.get_state_w_attribute(_init_state, "list_ap")
    except KeyError:
        _init_conf = two_player_game.trans_sys.get_state_w_attribute(_init_state, "ap")

    # transition system nodes and edges
    if transition_system is not None:
        _trans_sys_nodes = len(transition_system.transition_system._graph.nodes())
        _trans_sys_edges = len(transition_system.transition_system._graph.edges())
    else:
        _trans_sys_nodes = _trans_sys_edges = None

    # product graph nodes and edges
    if dfa_game:
        _prod_nodes = len(dfa_game._graph.nodes())
        _prod_edges = len(dfa_game._graph.edges())
    else:
        _prod_nodes = len(two_player_game.two_player_game._graph.nodes())
        _prod_edges = len(two_player_game.two_player_game._graph.edges())

    if regret_graph_of_alternatives is not None:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = len(regret_graph_of_alternatives._graph.nodes())
        _graph_of_alts_edges = len(regret_graph_of_alternatives._graph.edges())
    else:
        # graph of alternatives nodes and edges
        _graph_of_alts_nodes = None
        _graph_of_alts_edges = None

    try:
        _ltl_formula = two_player_game.formula
    except AttributeError:
        _ltl_formula = two_player_game._auto_graph._formula

    # create a data dict to dump it
    data_dict: Dict = dict(
        task_name=_task_name,
        no_of_boxes={
            'num': len(_boxes) if _boxes is not None else None,
            'objects': _boxes,
        },
        no_of_loc={
            'num': len(_locations) if _locations is not None else None,
            'objects': _locations
        },
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

    if admissible and minigrid:

        data_dict: Dict = dict(
            task_name=_task_name,
            no_of_boxes={
                'num': len(_boxes) if _boxes is not None else None,
                'objects': _boxes,
            },
            no_of_loc={
                'num': len(_locations) if _locations is not None else None,
                'objects': _locations
            },
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
            adv_val=_reg_value,
            adm_str=pos_seq
        )

    # now dump the data in a file
    if adversarial:
        _file_name: str = \
            f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_h_" \
            f"{_reg_value}_adv_"
    elif admissible and not minigrid:
        _file_name: str = \
            f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc__adm_"
    elif admissible and minigrid:
        _file_name: str = \
            f"/saved_strs/{_task_name}__adm_"
    elif minigrid and not minigrid:
        _file_name: str = \
            f"/saved_strs/{_task_name}__adv_"
    else:
        _file_name: str =\
        f"/saved_strs/{_task_name}_{len(_boxes)}_box_{len(_locations)}_loc_h_" \
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


def construct_abstraction(abstraction_instance: str,
                          print_flag: bool = False,
                          record_flag: bool = False,
                          render_minigrid: bool = False,
                          test_all_str: bool = False,
                          rollout_flag: bool = False,
                          max_iterations: int = 100,
                          human_type: str = '',
                          strategy_type: str = '',
                          env_type: str = '') -> None:
    """
    A function that will construct call the correct. Currently, we support Non-deterministic Manipulator and Minigrid instances . 

    Set test_all_str to True to test all strategy synthesis algorithms and all types of rollouts
    """
    
    if abstraction_instance not in VALID_ABSTRACTION_INSTANCES:
        warnings.warn(f"[Error] Please enter a valid Abstraction type:[ {', '.join(VALID_ABSTRACTION_INSTANCES)} ]")
        sys.exit(-1)
    
    if abstraction_instance == 'daig-main':
        daig_main(print_flag=print_flag, record_flag=record_flag, test_all_str=test_all_str, rollout_flag=rollout_flag)
    elif abstraction_instance == 'arch-main':
        arch_main(print_flag=print_flag, record_flag=record_flag, test_all_str=test_all_str)
    elif abstraction_instance == 'minigrid':
        minigrid_main(debug=print_flag, record=record_flag, render=render_minigrid, test_all_str=test_all_str, max_iterations=max_iterations, human_type=human_type, strategy_type=strategy_type, env_type=env_type)
    elif abstraction_instance == 'tic-tac-toe':
        tic_tac_toe_main(rollout_flag=rollout_flag, print_flag=print_flag)


def minigrid_main(debug: bool = False,
                  render: bool = False,
                  record: bool = False,
                  test_all_str: bool = False,
                  max_iterations: int = 100,
                  human_type: str = '',
                  strategy_type: str = '',
                  env_type: str = '') -> None:
    """
    Function that constructs the minigrid instances, constructs a product graph and rolls out a strategy.

    Currently supported envs
    nd_minigrid_envs = {'MiniGrid-FloodingLava-v0', 'MiniGrid-CorridorLava-v0', 'MiniGrid-ToyCorridorLava-v0',
        'MiniGrid-FishAndShipwreckAvoidAgent-v0', 'MiniGrid-ChasingAgentIn4Square-v0'}
    """
    # nd_minigrid_envs = ['MiniGrid-FloodingLava-v0', 'MiniGrid-CorridorLava-v0', 'MiniGrid-ToyCorridorLava-v0',
    #     'MiniGrid-FishAndShipwreckAvoidAgent-v0', 'MiniGrid-ChasingAgentIn4Square-v0', 'MiniGrid-FourGrids-v0', 
    #     'MiniGrid-ChasingAgent-v0', 'MiniGrid-ChasingAgentInSquare4by4-v0', 'MiniGrid-ChasingAgentInSquare3by3-v0']
    # nd_minigrid_envs = ['MiniGrid-IntruderRobotRAL25-v0']
    # nd_minigrid_envs = ['MiniGrid-LavaAdm_karan-v0', 'MiniGrid-IntruderRobotRAL25-v0', 'MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0']
    nd_minigrid_envs = ['MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0']
    # nd_minigrid_envs = ['MiniGrid-ThreeDoorIntruderRobotRAL25-v0']
    start = time.time()
    for id in nd_minigrid_envs:
        minigrid_handle = NonDeterministicMiniGrid(env_id=id,
                                                #    formula='!(agent_blue_right) U (floor_green_open)',
                                                   formula=minigrid_env_formulas[id],
                                                   player_steps = {'sys': [1], 'env': [1]},
                                                   save_flag=True,
                                                   env_snap_format='pdf',
                                                   env_dpi =500,
                                                   plot_minigrid=False,
                                                   plot_dfa=False,
                                                   plot_product=False,
                                                   debug=debug)
        
        # now construct the abstraction, the dfa and take the product
        
        if id in ['MiniGrid-FourDoorIntruderRobotCarpetRAL25-v0', 'MiniGrid-ThreeDoorIntruderRobotRAL25-v0', 'MiniGrid-IntruderRobotRAL25-v0']:
            minigrid_handle.build_minigrid_game(env_snap=False,
                                                only_augment_obs=False,
                                                modify_intruder_game=True,
                                                config_yaml_dict=OrderedDict(door_dict[id]))
        else:
            minigrid_handle.build_minigrid_game(env_snap=False, get_aps=False)
        # sys.exit(-1)
        minigrid_handle.get_aps(print_flag=True)
        # minigrid_handle.get_minigrid_edge_weights(print_flag=False)
        print(f"Sys Actions: {minigrid_handle.minigrid_sys_action_set}")
        print(f"Env Actions: {minigrid_handle.minigrid_env_action_set}")
        # minigrid_handle.modify_four_rooms_game(minigrid_handle.two_player_trans_sys, top_left_room=(1, 1), room_size=3)
        # InteractiveGraph.visualize_game(minigrid_handle.two_player_trans_sys, depth_limit=5)
        minigrid_handle.set_edge_weights(print_flag=False)
        minigrid_handle.build_automaton(ltlf=True)
        minigrid_handle.build_product()
        # modify the product game if its robot_evasion example - testing - the intruder can only move if the agent observes it.
        # minigrid_handle.modify_robot_evasion_game()
        # remove_non_reachable_states(game=minigrid_handle.dfa_game, debug=False)
        end = time.time()
        print(f"Done Constrcuting the DFA Game: {end-start:0.2f} seconds")
        print(f"No. of nodes in the product graph is :{len(minigrid_handle.dfa_game._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(minigrid_handle.dfa_game._graph.edges())}")
        
        # run all synthesins and rollout algorithms0
        if test_all_str:
            run_all_synthesis_and_rollouts(game=minigrid_handle.dfa_game,
                                        debug=False)
        
        # synthesize a strategy 
        # valid human types "manual", "no-human", "random-human", "epsilon-human", "coop-human", "mixed-human"
        # use "random-human" for Adv. Human
        # use "epsilon-human" with epsilon set to 1 for completely random human
        # use "coop-human" for cooperative human
        # use "mixed-human" for Adv. and Random Human
        else:
            str_handle, roller = run_synthesis_and_rollout(strategy_type=strategy_type,
                                                           #   strategy_type=VALID_STR_SYN_ALGOS[-2],
                                                           game=minigrid_handle.dfa_game,
                                                           #   human_type='manual',
                                                           human_type=human_type,
                                                           #   human_type ='epsilon-human',
                                                           #   human_type='random-human',
                                                           #    human_type ='coop-human',
                                                           #  human_type='mixed-human',
                                                           #  sys_type = 'random-sys',
                                                           rollout_flag=True,
                                                           epsilon=1,
                                                           debug=True,
                                                           max_iterations=max_iterations)
        
        minigrid_handle._logger._results.append(str_handle._logger.package_data())
        minigrid_handle._logger._episode += 1

        # run the simulation if the render or record flag is true
        if render or record:
            system_actions, env_actions = minigrid_handle._action_parser(action_seq=roller.action_seq)

            minigrid_handle.simulate_strategy(sys_actions=system_actions, env_actions=env_actions, render=render, record_video=record)
    
        minigrid_handle._logger.dump_results_to_yaml(file_path=ROOT_PATH + BENCHMARK_DIR + "/comp_time", add_time_stamp=True)
        

    # _dump_strs = input("Do you want to save the rollout of the strategy,Enter: Y/y")
    # # save strs
    # if _dump_strs == "y" or _dump_strs == "Y":
    #     save_str(admissible=True,
    #              minigrid=True,
    #              two_player_game=minigrid_handle.dfa_game,
    #              dfa_game=minigrid_handle.dfa_game,
    #              pos_seq=roller.action_seq,
    #              causal_graph=None,
    #              transition_system=None,
    #              adversarial=False)


@timer_decorator
def tic_tac_toe_main(rollout_flag: bool = False, print_flag: bool = False) -> None:
    game_handle = TicTacToe()
    game_handle.construct_graph(modify_weights=False)

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(game_handle._two_player_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(game_handle._two_player_game._graph.edges())}")

    dfa = game_handle.build_LTLf_automaton(formula=FORMULA_ADM_TIC_TAC_TOE_2)
    product_graph = game_handle.build_product(dfa=dfa, trans_sys=game_handle.two_player_game)

    if print_flag:
        print(f"No. of nodes in the DFA game is :"
              f"{len(product_graph._graph.nodes())}")
        print(f"No. of edges in the DFA game is :"
              f"{len(product_graph._graph.edges())}")

    _, roller = run_synthesis_and_rollout(strategy_type=VALID_STR_SYN_ALGOS[-1],
                                              game=product_graph,
                                            #   human_type='random-human',
                                              human_type='manual',
                                              rollout_flag=rollout_flag,
                                              debug=True,
                                              max_iterations=100,
                                              reg_factor=1.25)


    # _dump_strs = input("Do you want to save the strategy,Enter: Y/y")
    # # save strs
    # if _dump_strs == "y" or _dump_strs == "Y":
    #     save_str(admissible=True,
    #              two_player_game=product_graph,
    #              pos_seq=roller.action_seq,
    #              causal_graph=None,
    #              transition_system=None,
    #              adversarial=False)


@timer_decorator
def daig_main(print_flag: bool = False, record_flag: bool = False, test_all_str: bool = False, rollout_flag: bool = False) -> None:
    # domain_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/domain.pddl"
    # _problem_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/problem.pddl"
    # problem_file_path = ROOT_PATH + "/pddl_files/two_table_scenario/diagonal/sym_test_problem.pddl"

    ##### Adm Related domain files #####
    domain_file_path = ROOT_PATH + '/pddl_files/adm_unrealizable_world/domain.pddl'
    # problem_file_path = ROOT_PATH + '/pddl_files/adm_unrealizable_world/problem.pddl'

    #### Safe-Adm game domain file - ICRA 25 ####
    # problem_file_path = ROOT_PATH + '/pddl_files/adm_unrealizable_world/problem_2.pddl'
    problem_file_path = ROOT_PATH + '/pddl_files/adm_unrealizable_world/problem_3.pddl'


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
    modify_abstraction(game=two_player_instance.two_player_implicit_game,
                       all_human_loc=set(two_player_instance.causal_graph.task_intervening_locations),
                       hopeless_human_loc=set(['l6', 'l7', 'l8']),
                       human_only_loc=set(['l9']),
                       debug=False)
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

    # print(f"# of Sys states in Two player game: {sys_count}")
    # print(f"# of Env states in Two player game: {env_count}")

    if print_flag:
        print(f"No. of nodes in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.nodes())}")
        print(f"No. of edges in the Two player game is :"
              f"{len(two_player_instance._two_player_implicit_game._graph.edges())}")
    # sys.exit(-1)
    # dfa = two_player_instance.build_LTL_automaton(formula=FORMULA_SAFE_ADM_TEST_2,  plot=True)
    dfa = two_player_instance.build_LTLf_automaton(formula=FORMULA_SAFE_ADM_TEST_2, plot=True)
    # sys.exit(-1)

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
    
    # create a strategy synthesis handle, solve the game, and roll out the strategy
    if test_all_str:
        run_all_synthesis_and_rollouts(game=product_graph,
                                       debug=False)
    else:    
        _, roller = run_synthesis_and_rollout(strategy_type=VALID_STR_SYN_ALGOS[-1],
                                              game=product_graph,
                                            #   human_type='random-human',
                                              human_type='manual',
                                              rollout_flag=rollout_flag,
                                              debug=True,
                                              max_iterations=100,
                                              reg_factor=1.25)

    # return
    # ask the user if they want to save the str or not
    # _dump_strs = input("Do you want to save the strategy,Enter: Y/y")
    # # save strs
    # if _dump_strs == "y" or _dump_strs == "Y":
    #     save_str(admissible=True,
    #              causal_graph=causal_graph_instance,
    #              transition_system=transition_system_instance,
    #              two_player_game=two_player_instance,
    #              pos_seq=roller.action_seq,
    #              adversarial=False)


@timer_decorator
def arch_main(print_flag: bool = False, record_flag: bool = False, test_all_str: bool = False) -> None:
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

    product_graph = two_player_instance.build_product(dfa=dfa,
                                                      trans_sys=two_player_instance.two_player_implicit_game)

    relabelled_graph = two_player_instance.internal_node_mapping(product_graph)

    if print_flag:
        print(f"No. of nodes in the product graph is :{len(relabelled_graph._graph.nodes())}")
        print(f"No. of edges in the product graph is :{len(relabelled_graph._graph.edges())}")

    # create a strategy synthesis handle, solve the game, and roll out the strategy
    if test_all_str:
        run_all_synthesis_and_rollouts(game=product_graph,
                                       debug=False)
    else:    
        _, roller = run_synthesis_and_rollout(strategy_type=VALID_STR_SYN_ALGOS[0],
                                              game=product_graph,
                                              human_type='no-human',
                                              rollout_flag=True,
                                              debug=True,
                                              max_iterations=100)

    # ask the user if they want to save the str or not
    _dump_strs = input("Do you want to save the strategy,Enter: Y/y")

    # save strs
    if _dump_strs == "y" or _dump_strs == "Y":
        save_str(causal_graph=causal_graph_instance,
                 transition_system=transition_system_instance,
                 two_player_game=two_player_instance,
                 pos_seq=roller.action_seq,
                 adversarial=True)


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
        # starting the monitor
        tracemalloc.start()
        construct_abstraction(abstraction_instance='minigrid',
                              print_flag=True,
                              record_flag=record,
                              render_minigrid=False,
                              test_all_str=False,
                              rollout_flag= True,
                              max_iterations=MAX_ITERATIONS,
                              human_type=args.human_type,
                              strategy_type=args.strategy_type,
                              env_type=args.env_type)

        # displaying the memory - output current memory usage and peak memory usage
        _,  peak_mem = tracemalloc.get_traced_memory()
        print(f" Peak memory [MB]: {peak_mem/(1024*1024)}")
        
        # stopping the library
        tracemalloc.stop()