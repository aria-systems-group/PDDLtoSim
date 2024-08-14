import os
import sys
import time
import gym
import pprint
import warnings
import itertools
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from gym_minigrid.minigrid import (MiniGridEnv, Grid, Lava, Floor,
                                   Ball, Key, Door, Goal, Wall, Box)

from gym_minigrid.register import register
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
from wombats.systems.minigrid import MultiAgentMiniGridEnv, ConstrainedAgent, Agent, MultiObjGrid, Carpet, Water


class NonDeterministicMiniGrid():

    def __init__(self,
                 env_id: str,
                 formula: str,
                 player_steps: Dict = {'sys': [1], 'env': [1]},
                 save_flag: bool = True,
                 plot_minigrid: bool = False,
                 plot_dfa: bool = False,
                 plot_product: bool = False,
                 env_snap_format: str = 'png',
                 env_dpi: float = 30,
                 debug: bool = False) -> None:
        
        self._available_envs: Dict = {}
        self.env_id: str = env_id
        self.minigrid_env: MultiAgentEnvType = gym.make(env_id)
        self.formula: str = formula
        # dictionary to allow multistep for agents in the env
        self.player_steps: Dict[str, List[int]] = player_steps

        self._two_player_trans_sys: Optional[TwoPlayerGraph] = None
        self._dfa: Optional[DFAGraph] = None
        self._dfa_game: Optional[ProductAutomaton] = None

        self._game_aps: Set[str] = set({})
        self._minigrid_edge_weights: Set[int] = set({})
        self._minigrid_sys_action_set  = set({})
        self._minigrid_env_action_set  = set({})

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
        if self._two_player_trans_sys is None:
            warnings.warn("[Error] Tried accessing the minigrid two player game abstraction without constucting the abstraction. \
                           Please run `build_minigrid_game()` function.")
            sys.exit(-1)
        return self._two_player_trans_sys
    
    @property
    def dfa(self):
        return self._dfa
    
    @property
    def dfa_game(self):
        return self._dfa_game
    
    @property
    def game_aps(self):
        if not bool(self._game_aps):
            self.get_aps(print_flag=False)
        return self._game_aps
    
    @property
    def minigrid_edge_weights(self):
        if not bool(self._minigrid_edge_weights):
            self.get_minigrid_edge_weights(print_flag=False)
        return self._minigrid_edge_weights
    

    @property
    def minigrid_sys_action_set(self):
        if not bool(self._minigrid_sys_action_set):
            self.get_minigrid_player_actions(print_flag=False)
        return self._minigrid_sys_action_set


    @property
    def minigrid_env_action_set(self):
        if not bool(self._minigrid_env_action_set):
            self.get_minigrid_player_actions(print_flag=False)
        return self._minigrid_env_action_set
    
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

        self._player_steps = step_dict
    

    def get_aps(self, print_flag: bool = False):
        """
        A helper method that returns the set of all Atomic Propositions (APs) in the Minigrid Two player abstraction.
          Used in writing formula as Formula is defiend over APs.
        """
        # set of valid attributes - ap, player, init
        for s in self.two_player_trans_sys._graph.nodes():
            self._game_aps.add(self.two_player_trans_sys.get_state_w_attribute(s, 'ap'))
        
        if print_flag:
            print(f"Set of APs: {self.game_aps}")

    def get_minigrid_edge_weights(self, print_flag: bool = False):
        """
        A helper method that returns the set of all edge weights in the Minigrid Two player abstraction.
        """
        for _s in  self.two_player_trans_sys._graph.nodes():
            for _e in  self._two_player_trans_sys._graph.out_edges(_s):
                if  self.two_player_trans_sys._graph[_e[0]][_e[1]][0].get('weight', None):
                    self._minigrid_edge_weights.add( self.two_player_trans_sys._graph[_e[0]][_e[1]][0]["weight"])
        
        if print_flag:
            print(f"Set of Org Minigrid Two player game edge weights: {self.minigrid_edge_weights}")
    

    def get_minigrid_player_actions(self, print_flag: bool = False) -> None:
        """
        A helper method that sets the sys_action_set and env_action_set attributes by parsing the org actions in the Minigrid Two player abstraction.

        The action set is stored a tuple of motion primitives. As minigrid is a gridworld, the most basic primitive are cardinal actions: North, South, East, and West.
        Tuple (north, north) means the player can travel north twice in one timestep.
        """
        for player, multiactions in self.minigrid_env.player_actions.items():
            for multiaction in multiactions:
                action_strings = []
                for agent, actions in zip(self.minigrid_env.unwrapped.agents, multiaction):
                    action_string = []
                    for action in actions:
                        if action is None or np.isnan(action):
                            continue
                        a_str = agent.ACTION_ENUM_TO_STR[action]
                        action_string.append(a_str)
                    action_strings.append(tuple(action_string))
                if player == 'sys':
                    self._minigrid_sys_action_set.add(tuple(action_strings[0]))
                elif player == 'env':
                    self._minigrid_env_action_set.add(tuple(action_strings[1:]))
        
        if print_flag:
            print(f"Set of Org Minigrid Two player game Sys Actions: {self._minigrid_sys_action_set}")
            print("****************************************************************")
            print(f"Set of Org Minigrid Two player game Env Actions: {self._minigrid_env_action_set}")
    

    def set_edge_weights(self, print_flag: bool = False):
        """
        A helper method that sets edge weights in the Minigrid Two player abstraction.
          Currently, only supports uniform edge weight of 1 for sys actions and 0 for env actions.

        # TODO: Update this to be more modular. Like action dependent costs, or state-action dependent costs.
        """
        for _s in self.two_player_trans_sys._graph.nodes():
            for _e in self.two_player_trans_sys._graph.out_edges(_s):
                self.two_player_trans_sys._graph[_e[0]][_e[1]][0]["weight"] = 1 if self.two_player_trans_sys._graph.nodes(data='player')[_s] == 'eve' else 0
        
        if print_flag:
            print("Done setting edge weights ")


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
                            'MiniGrid-FishAndShipwreckAvoidAgent-v0': [], 'MiniGrid-ChasingAgentIn4Square-v0': [],
                            'MiniGrid-FourGrids-v0': [], 'MiniGrid-ChasingAgent-v0': [], 'MiniGrid-ChasingAgentInSquare4by4-v0': [],
                            'MiniGrid-ChasingAgentInSquare3by3-v0': [], 'MiniGrid-LavaComparison_karan-v0': [], 'MiniGrid-LavaAdm_karan-v0': []}

        self._available_envs = nd_minigrid_envs


    def build_automaton(self, use_alias: bool = False, ltlf: bool = True):
        """
        A method to construct automata using the regret_synthesis_tool.
        """
        if ltlf:
            ltl_automaton = graph_factory.get('LTLfDFA',
                                              graph_name="minigrid_ltl",
                                              config_yaml="/config/minigrid_ltl",
                                              save_flag=self.save_flag,
                                              ltlf=self.formula,
                                              plot=self.plot_dfa)
        else:
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
                                               absorbing=False,
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


    def build_minigrid_game(self,
                            env_snap: bool = False,
                            get_aps: bool = False,
                            get_weights: bool = False,
                            set_weights: bool = False):
        """
         Build OpenAI Minigrid Env with multiple agents, then construct the corresponding graph.
        """
        self.minigrid_env = DynamicMinigrid2PGameWrapper(self.minigrid_env,
                                           player_steps=self.player_steps,
                                           monitor_log_location=os.path.join(DIR, GYM_MONITOR_LOG_DIR_NAME))
        self.minigrid_env.reset()
        env_filename = os.path.join(DIR, 'plots', f'{self.env_id}_Game.png')
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

        # get all the aps, and the player
        if get_aps:
            self.get_aps(debug=self.debug)
        
        # get edge weight sets
        if get_weights:
            self.get_org_edge_weights(debug=self.debug)
        
        # set edge weight sets
        if set_weights:
            self.set_edge_weights(debug=self.debug)

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
                    render=render,
                    record_video=record_video)
        else:
            sim.run(iterations=iterations,
                    sys_actions=sys_actions,
                    env_strategy='random',
                    render=render,
                    record_video=record_video)
        sim.get_stats()
    

    def _action_parser(self, action_seq: List[str]) -> Tuple[List[Tuple[str]]]:
        """
        A function that paerse the sequence of actions from the rollout provider class an converts into suitable format
          for the simulator function.
        
          For exmaple, for one sys agent and one env agent, the actions are of the form (north__None) and for multiactions they are of the form (north_north__None)
          The `__` is the delimiter for the two agents.

          This function parses the string and returns a List of tuples of actions for sys and env agents. e.g., (`north`) or (`north`, `north`)

          For, one sys agent and multiple env agents, the actions are concatenated with `__` as delimiter. e.g., `None__north__None` and `None__None__south`. 
          As, it is turn-based game, one of the env agent moves at a given instance. 

          # TODO: Update this function to handle multiple env agents
          # TODO: Update this in Wombats, to allow all the env agents to move at the same time. 
        """
        system_actions: List[tuple] = []
        env_actions:List[tuple] = []

        # manually parse the actions, the first action is by Sys player, the next one is by Env ...
        for itr, act in enumerate(action_seq):
            if itr % 2 == 0:
                assert act.split('__')[1] == 'None', "Error when rolling out strategy"
                # action edge is of type North_North__None
                sys_action = act.split('__')[0]
                act_tuple = tuple(sys_action.split('_'))
                system_actions.append(act_tuple)
                
            elif itr % 2 != 0:
                assert act.split('__')[0] == 'None', "Error when rolling out strategy"
                # action edge is of type None__South_South
                env_action = act.split('__')[1]
                act_tuple = tuple(env_action.split('_'))
                env_actions.append(act_tuple)
        
        return system_actions, env_actions


class CorridorLava(MultiAgentMiniGridEnv):
    """
    """

    def __init__(
        self,
        width=11,
        height=8,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        # env_agent_start_pos=[(3, 2), (7, 2)],
        # env_agent_start_dir=[0, 0],
        env_agent_start_pos=[(8, 5)],
        env_agent_start_dir=[0],
        goal_pos=[(8, 6)],
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir

        self.goal_pos = goal_pos

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # self.grid.wall_rect(3, 4, 5, 1)
        # self.put_obj(Wall(), *(3, 5))
        # self.put_obj(Wall(), *(7, 5))
        # self.put_obj(Wall(), *(5, 6))

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        # self.put_obj(Water(), 3, 6)
        # self.put_obj(Water(), 4, 6)
        # self.put_obj(Water(), 4, 5)
        # self.put_obj(Water(), 5, 5)
        # self.put_obj(Water(), 6, 5)
        # self.put_obj(Water(), 6, 6)
        # self.put_obj(Water(), 7, 6)

        # bottom carpet
        # self.put_obj(Carpet(), 1, 2)
        # self.put_obj(Carpet(), 1, 3)
        # self.put_obj(Carpet(), 1, 4)
        # self.put_obj(Carpet(), 9, 2)
        # self.put_obj(Carpet(), 9, 3)
        # self.put_obj(Carpet(), 9, 4)
        # self.put_obj(Carpet(), 9, 5)

        # put lava around the goal region
        self.put_obj(Lava(), 7, 6)
        self.put_obj(Lava(), 9, 6)
        self.put_obj(Lava(), 7, 5)
        self.put_obj(Lava(), 9, 5) 

        # Lava
        self.put_obj(Lava(), 3, 3)
        self.put_obj(Lava(), 4, 3)
        self.put_obj(Lava(), 5, 3)
        self.put_obj(Lava(), 6, 3)
        self.put_obj(Lava(), 7, 3)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
        else:
            self.place_agent()

        for i in range(len(self.env_agent_start_pos)):
            p = self.env_agent_start_pos[i]
            d = self.env_agent_start_dir[i]
            # restricted_positions = [(i+1, j+1) for i, j in itertools.product(range(8), range(3, 8))]
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    restricted_objs=['lava', 'carpet', 'water', 'floor'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal squares, don"t touch lava'
    


class CorridorLavaAAAI25(MultiAgentMiniGridEnv):
    """
        Create a minigrid env. 5 by 5 gridworld with Goal on top left. Sys agent start on thr bottom left and Env agent starts on the top right.
        A winning stratgey does not exists. 
    """

    def __init__(
        self,
        width=6,
        height=6,
        agent_start_pos=(1, 4),
        agent_start_dir=0,
        env_agent_start_pos=[(4, 1)],
        env_agent_start_dir=[0],
        goal_pos=[(1, 1)],
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir

        self.goal_pos = goal_pos

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place a goal square in the top-left corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
        else:
            self.place_agent()

        for i in range(len(self.env_agent_start_pos)):
            p = self.env_agent_start_pos[i]
            d = self.env_agent_start_dir[i]
            # restricted_positions = [(i+1, j+1) for i, j in itertools.product(range(8), range(3, 8))]
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    restricted_objs=['lava', 'carpet', 'water', 'floor'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal squares, don"t touch lava'
    


register(
    id='MiniGrid-LavaComparison_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:CorridorLava'
)

register(
    id='MiniGrid-LavaAdm_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:CorridorLavaAAAI25'
)