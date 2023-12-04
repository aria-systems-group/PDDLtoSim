import os
import sys
import time
import gym
import pprint
import warnings

from pathlib import Path
from typing import Dict, List, Optional

from gym.envs.registration import registry

from regret_synthesis_toolbox.src.simulation.simulator import Simulator
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph, DFAGraph, ProductAutomaton
from regret_synthesis_toolbox.src.graph import graph_factory
from utls import is_docker

# set path constants
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = ROOT_PATH

# add wombats to my sys path
sys.path.append(f"{ROOT_PATH}/regret_synthesis_toolbox")

from wombats.systems.minigrid import DynamicMinigrid2PGameWrapper, GYM_MONITOR_LOG_DIR_NAME, MultiAgentEnvType, MultiStepMultiAgentAction


class NonDeterministicMiniGrid():

    def __init__(self,
                 env_id: str,
                 formula: str,
                 save_flag: bool = True,
                 plot_minigrid: bool = False,
                 plot_dfa: bool = False,
                 plot_product: bool = False,
                 env_snap_format: str = 'png',
                 env_dpi: float = 300,
                 debug: bool = False) -> None:
        
        self._available_envs: Dict = {}
        self.env_id: str = env_id
        self.minigrid_env: MultiAgentEnvType = gym.make(env_id)
        self.formula: str = formula
        # dictionary to allow multistep for agents in the env
        self._player_steps: Dict[str, List[int]] = {'sys': [1], 'env': [1]}

        self._two_player_trans_sys: Optional[TwoPlayerGraph] = None
        self._dfa: Optional[DFAGraph] = None
        self._dfa_game: Optional[ProductAutomaton] = None

        # saving yaml files related to graphs
        self.save_flag: bool = save_flag

        # plotting related attributes
        self.plot_minigrid: bool = plot_minigrid
        self.plot_dfa: bool = plot_dfa
        self.plot_product: bool = plot_product
        self.debug: bool = debug

        # env snaprshot related attributes
        self.env_snap_format = env_snap_format
        self.env_dpi = env_dpi

        # now construct the abstraction, the dfa and take the product
        self.build_minigrid_game()
        self.build_automaton()
        self.build_product()
    

    @property
    def env_id(self):
        return self._env_id

    @property
    def formula(self):
        return self._formula
    
    @property
    def player_steps(self):
        return self._player_steps
    
    @property
    def two_player_trans_sys(self):
        return self._two_player_trans_sys
    
    @property
    def dfa(self):
        return self._dfa
    
    @property
    def dfa_game(self):
        return self._dfa_game
    
    @property
    def available_envs(self):
        if not bool(self._available_envs):
            self.get_available_non_det_minigrid_envs(print_envs=False)

        return self._available_envs

    @env_id.setter
    def env_id(self, minigrid_id: str):
        if minigrid_id not in self.available_envs.keys():
            warnings.warn(f"[Error] Please enter a valid env from: [ {', '.join(self.available_envs.keys())} ]")
            sys.exit(-1)
        
        self._env_id = minigrid_id
    
    @formula.setter
    def formula(self, formula_str: str):
        if not isinstance(formula_str, str):
            warnings.warn(f"[Error] Please enter formula as string. Currently it is of type: {type(formula_str)}.")
            sys.exit(-1)

        self._formula = formula_str
    

    @player_steps.setter
    def player_steps(self, step_dict: str):
        if not isinstance(step_dict, dict):
            warnings.warn(f"[Error] Please enter player steps as dict. Currently it is of type: {type(step_dict)}.")
            sys.exit(-1)
        
        assert len(step_dict.keys()) == 2 and ('sys' in step_dict.keys() and 'env' in step_dict.keys()), "The player step size diction should of type \{'sys': [1], 'env': [1]\}"

        self._formula = step_dict


    def get_available_non_det_minigrid_envs(self, print_envs: bool = False) -> Dict:
        """
        A function that  returns a list of avilable envs
        """
        minigrid_envs = {env: v for env, v in registry.env_specs.items() if isinstance(env, str) and env.startswith('MiniGrid-')}
        if print_envs:
            print("All MiniGrid Envs")
            pprint.pprint(minigrid_envs.keys())
        
        nd_minigrid_envs = {env: v for env, v in minigrid_envs.items() if v.nondeterministic}

        # TODO: Update Wombats - update nondeterministic attribute to True for Non-Deterministic Envs. 
        if print_envs:
            print("All Non-Deterministic MiniGrid Envs")
            pprint.pprint(nd_minigrid_envs)
        
        # for now we will override this with hardcoded values=
        # a few more that need to tested 'MiniGrid-FourGrids-v0', 'MiniGrid-ChasingAgent-v0', 'MiniGrid-ChasingAgentInSquare4by4-v0', 'MiniGrid-ChasingAgentInSquare3by3-v0'
        nd_minigrid_envs = {'MiniGrid-FloodingLava-v0': [], 'MiniGrid-CorridorLava-v0': [], 'MiniGrid-ToyCorridorLava-v0': [], 
                            'MiniGrid-FishAndShipwreckAvoidAgent-v0': [], 'MiniGrid-ChasingAgentIn4Square-v0': []}

        self._available_envs = nd_minigrid_envs


    def build_automaton(self, use_alias: bool = False):
        """
        A method to construct automata using the regret_synthesis_tool.
        """
        ltl_automaton = graph_factory.get('DFA',
                                            graph_name="minigrid_ltl",
                                            config_yaml="/config/minigrid_ltl",
                                            save_flag=self.save_flag,
                                            sc_ltl=self.formula,
                                            use_alias=use_alias,
                                            plot=self.plot_dfa)

        if self.debug:
            print(f"The pddl formula is : {self.formula}")
        

        self._dfa = ltl_automaton


    def build_product(self):
        product_automaton = graph_factory.get("ProductGraph",
                                               graph_name="minigrid_product_graph",
                                               config_yaml="/config/minigrid_product_graph",
                                               trans_sys=self.two_player_trans_sys,
                                               automaton=self.dfa,
                                               save_flag=self.save_flag,
                                               prune=False,
                                               debug=False,
                                               absorbing=True,
                                               finite=False,
                                               plot=self.plot_product,
                                               pdfa_compose=False)

        print("Done building the Product Automaton")

        self._dfa_game = product_automaton


    def initialize_edge_labels_on_fancy_graph(self, two_player_graph):
        """
         This method post process the abstraction. The Original abstraction can contain multiple edged from the same state to the successor state.
           My current product construction(w pdfa_composeflag set to False) and synthesis can not accomodate multiple edges between same states. 
        
           TODO: Add this as an improvement to Regret_synthesis src code in future iterations. 
        """

        edges = set(two_player_graph._graph.edges())

        for edge in edges:
            for i, edge_data in two_player_graph._graph[edge[0]][edge[1]].items():
                actions = edge_data.get('symbols')
                weights = edge_data.get('weight')
                two_player_graph._graph[edge[0]][edge[1]][i]['actions'] = actions[0]
                two_player_graph._graph[edge[0]][edge[1]][i]['weight'] = weights[0]


    def build_minigrid_game(self, env_snap: bool = False):
        """
         Build OpenAI Minigrid Env with multiple agents, then construct the corresponding grapgh
        """
        self.minigrid_env = DynamicMinigrid2PGameWrapper(self.minigrid_env,
                                           player_steps=self.player_steps,
                                           monitor_log_location=os.path.join(DIR, GYM_MONITOR_LOG_DIR_NAME))
        self.minigrid_env.reset()
        env_filename = os.path.join(DIR, 'plots', 'gym_env.png')
        Path(os.path.split(env_filename)[0]).mkdir(parents=True, exist_ok=True)
        if env_snap:
            self.minigrid_env.render_notebook(env_filename, self.env_dpi)

        file_name = self.env_id + 'Game'
        filepath = os.path.join(DIR, 'config', file_name)
        config_yaml = os.path.relpath(filepath, ROOT_PATH)

        # Game Construction
        start = time.time()
        two_player_graph = graph_factory.get('TwoPlayerGraph',
                                             graph_name='TwoPlayerGame',
                                             config_yaml=config_yaml,
                                             from_file=False,
                                             minigrid=self.minigrid_env,
                                             save_flag=self.save_flag,
                                             plot=self.plot_minigrid,
                                             view=False,
                                             format=self.env_snap_format)
        end = time.time()
        print(f"Two Player Graph Construction (s): {end - start}")

        # add labels to the graph and prune multiple edges from same state to successor state
        self.initialize_edge_labels_on_fancy_graph(two_player_graph)

        self._two_player_trans_sys = two_player_graph
    

    def simulate_strategy(self,
                          sys_actions: MultiStepMultiAgentAction,
                          env_actions: Optional[MultiStepMultiAgentAction] = None,
                          iterations: int = 100,
                          render: bool = False,
                          record_video: bool = False):
        """
        A function to simulate the synthesized strategy. Available env_action 'random' and 'interactive'.

        sys_action: is a list of system actions as string, e.g. ['north__None', ...]
        env_action: is a list of env actions as string, e.g. ['None__south', ...] 
        """
        sim = Simulator(self.minigrid_env, self.dfa_game)

        # turn of rendering if running inside docker
        if render and is_docker():
            render = False

        if env_actions is not None:
            sim.run(iterations=iterations,
                    sys_actions=sys_actions,
                    env_actions=env_actions,
                    env_strategy=env_actions if env_actions is not None else 'random',
                    render=render,
                    record_video=record_video)
        else:
            sim.run(iterations=iterations,
                    sys_actions=sys_actions,
                    env_strategy='random',
                    render=render,
                    record_video=record_video)
        sim.get_stats()
