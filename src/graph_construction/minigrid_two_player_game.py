import os
import sys
import time
import gym
import pprint
import warnings
import itertools
import numpy as np

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from gym_minigrid.minigrid import (MiniGridEnv, Grid, Lava, Floor,
                                   Ball, Key, Door, Goal, Wall, Box)

from gym_minigrid.register import register
from gym.envs.registration import registry

from icra_examples.safe_adm_game import remove_non_reachable_states
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

TS_INIT_STATE_NAME = 'Init'


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
                            'MiniGrid-ChasingAgentInSquare3by3-v0': [], 'MiniGrid-LavaComparison_karan-v0': [], 'MiniGrid-LavaAdm_karan-v0': [],
                            'MiniGrid-NarrowLavaAdm_karan-v0': [], 'MiniGrid-IntruderRobotRAL25-v0': [], 'MiniGrid-TwoNarrowLavaAdm_karan-v0': [],
                            'MiniGrid-FourRoomsRobotRAL25-v0': [], 'MiniGrid-SmallFourRoomsRobotRAL25-v0': []}

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
                                               observe_next_on_trans=True,
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
                            set_weights: bool = False,
                            augment_obs: bool = False):
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
                                             minigrid_wait=False,
                                             save_flag=self.save_flag,
                                             plot=self.plot_minigrid,
                                             view=False,
                                             format=self.env_snap_format)
        end = time.time()
        print(f"Two Player Graph Construction (s): {end - start}")

        # add labels to the graph and prune multiple edges from same state to successor state
        self.initialize_edge_labels_on_fancy_graph(two_player_graph)

        self._two_player_trans_sys = two_player_graph

        if augment_obs:
            self._augment_obs()    

        # get all the aps, and the player
        if get_aps:
            self.get_aps(print_flag=True)
        
        # get edge weight sets
        if get_weights:
            self.get_org_edge_weights(debug=self.debug)
        
        # set edge weight sets
        if set_weights:
            self.set_edge_weights()
        
        # sys.exit(-1)
    

    def modify_robot_evasion_game(self):
        """
         A heleper fuction to remove Env agent action to move if Sys agent has not observed it yet.
        """
        # remove Eve's action if the DFA state is in the init state.
        dfa_init = self.dfa.get_initial_states()[0][0]  # should q1
        env_edges_to_rm = set()
        for _s in self.dfa_game._graph.nodes():
            # make an exception for Env state to go north (self-loop) so that there exisits a path on the DFA game from q1 to other dfa states.
            if self.dfa_game._graph.nodes(data='player')[_s] == 'adam' and _s[1] == dfa_init:
                for _e in self.dfa_game._graph.out_edges(_s):
                    if isinstance(_e[1], tuple) and _e[0][0][2][0] == (8, 1) and _e[1][0][2][0] == (8, 1): 
                        continue
                    env_edges_to_rm.add(_e)

        self.dfa_game._graph.remove_edges_from(env_edges_to_rm)
        

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
    
    def _augment_obs(self):
        """
         A helper method used to augment the AP at each state and see if the Agent observed the Env player or not.
        """
        for state in self._two_player_trans_sys._graph.nodes():
            # augement ap with agent_observed label
            agent_pos = state[2][0]
            sys_pos = state[1][0]
            if (0 <= agent_pos[0] - sys_pos[0] <= 2) and (0 <= sys_pos[1] - agent_pos[1] <= 2):
                if self._two_player_trans_sys._graph.nodes[state].get('ap') == '':
                    self._two_player_trans_sys._graph.nodes[state]['ap'] = "agent_observed"
                # the wall are not see thought. So if you are in the human room, you cannot observe the agent. Further, avoid creating labels where you are in collision with the agent
                elif self._two_player_trans_sys._graph.nodes[state]['ap'] not in ["floor_purple_open", "agent_blue_right"] :
                    self._two_player_trans_sys._graph.nodes[state]['ap'] = self._two_player_trans_sys._graph.nodes[state].get('ap') + \
                        "__" + "agent_observed"


    def modify_four_rooms_game(self,
                               game: TwoPlayerGraph,
                               top_left_room: tuple,
                               room_direction: Optional[dict] = None,
                               room_size: Optional[int] = None,
                               corridor_size: Optional[int] = None):
        """
         A helper fuction to modify the FourRooms game to make it more interesting.
        """
        modify_handle = ModifiedFourRooms2PGame(game,
                                                top_left_room,
                                                room_size=room_size, 
                                                room_direction={'r1': ModifiedFourRooms2PGame.Direction.ANTICLOCKWISE, 
                                                                'r2': ModifiedFourRooms2PGame.Direction.CLOCKWISE,
                                                                'r3': ModifiedFourRooms2PGame.Direction.CLOCKWISE,
                                                                'r4': ModifiedFourRooms2PGame.Direction.ANTICLOCKWISE},
                                                minigrid_env=self.minigrid_env,
                                                fake_init_state=True)
        modify_handle.modify_four_rooms_game(debug=True)
    

class ModifiedFourRooms2PGame:

    class Direction(Enum):
        CLOCKWISE = 1
        ANTICLOCKWISE = -1

    class Room:
        def __init__(self, name: str, x: int, y: int, size: int):
            self.name = name
            self.x = x  # top-left corner x coordinate
            self.y = y  # top-left corner y coordinate
            self.size = size  # size of the room (assuming square)

            # clockwise from top-left
            self.corners = [
                (x, y),
                (x + self.size - 1, y),
                (x + self.size - 1, y + (self.size - 1)),
                (x, y + (self.size - 1))]

        def is_inside(self, pos: tuple):
            """
            A helper function to check if the agent is inside the room or not.
            """
            x, y = pos[0], pos[1]
            return (self.corners[0][0] <= x <= self.corners[1][0] and  # within x bounds
                self.corners[0][1] <= y <= self.corners[3][1])  # within y bounds

    def __init__(self,
                game: TwoPlayerGraph,
                room_2: tuple,
                room_direction: dict= {'r1': Direction.ANTICLOCKWISE ,
                                       'r2': Direction.CLOCKWISE,
                                       'r3': Direction.ANTICLOCKWISE,
                                       'r4': Direction.CLOCKWISE},
                room_size: int = 5,
                corridor_size: int = 1,
                minigrid_env: Optional[DynamicMinigrid2PGameWrapper] = None,
                fake_init_state: bool = False):
        self._game = game
        self._room_2 = self.Room('r2', room_2[0], room_2[1], room_size)
        self._room_1 = self.Room('r1', room_2[0] + room_size + corridor_size + 2, room_2[1], room_size)
        self._room_3 = self.Room('r3', room_2[0], room_2[1] + (room_size + corridor_size + 2), room_size)
        self._room_4 = self.Room('r4', room_2[0] + room_size + corridor_size + 2, room_2[1] + (room_size + corridor_size + 2), room_size)
        self._rooms = [self._room_1, self._room_2, self._room_3, self._room_4]
        self.allowed_moves: dict = {'clockwise' : {'top': (1, 0), 'bottom': (-1, 0), 'left': (0, -1), 'right': (0, 1)},
                                    'counterclockwise' : {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, 1), 'right': (0, -1)}}
        self.room_direction = room_direction
        if fake_init_state:
            assert minigrid_env is not None, "[Error] Please provide the minigrid env object to create the fake init state."
            self.minigrid_env = minigrid_env
            self._extend_trans_init()
            
    def prune_edge_based_on_direction(self, agent_pos: tuple, agent_fwd_pos: tuple, debug: bool = False, agent_info: str = '') -> bool:
        """
         A heleper fuction to implement an ahgent going with clockwise or counter clockwise direction in each of the four rooms.
        """
        # determine which room the agent is in
        for room in self._rooms:
            if room.is_inside(agent_pos):
                agent_room = room
                break
        else:
            # if debug:
            #     print(f"Agent pos {agent_pos} in not in any room ")
            return False

        # check if the next state is outside a room. If yes, then do not prune the edge.
        for room in self._rooms:
            if room.is_inside(agent_fwd_pos):
                break
        else:
            return False

        # get the allowed direction if this room
        if self.room_direction[agent_room.name].value == 1:
            local_allowed_moves: dict = self.allowed_moves['clockwise']
        else:
            local_allowed_moves: dict = self.allowed_moves['counterclockwise']
        
        # check if the agent is on the top 
        x, y = agent_pos[0], agent_pos[1]
        if agent_room.corners[0][0] <= x < agent_room.corners[1][0] and y == agent_room.corners[0][1]:
            relative_pos = 'top'
        elif agent_room.corners[0][0] < x <= agent_room.corners[1][0] and y == agent_room.corners[2][1]:
            relative_pos = 'bottom'
        elif agent_room.corners[1][0] == x and agent_room.corners[1][1] <= y < agent_room.corners[2][1]:
            relative_pos = 'right'
        elif agent_room.corners[3][0] == x and agent_room.corners[0][1] < y <= agent_room.corners[3][1]:
            relative_pos = 'left'
        else:
            warnings.warn['[Error] Agent is not in any of the corridors within the room. FIX THIS!!!!']
        
        # if debug:
        #     print(f"Agent Pos {agent_pos} belongs to room {agent_room.name} and relative pos is {relative_pos}")
        
        # check the move is legal as per the direction or not
        dx, dy = agent_fwd_pos[0] - agent_pos[0], agent_fwd_pos[1] - agent_pos[1]
        edge_direction = (dx, dy)

        if edge_direction != local_allowed_moves[relative_pos]:
            
            if debug:
                print(f"Removing edge {agent_info}: {agent_pos} ---> {agent_fwd_pos}")
            return True
        
        return False

    def modify_four_rooms_game(self, debug: bool = False):
        """
         A helper fuction to modify the FourRooms game to make it more interesting.
        """
        # get the agent positions
        edges_to_remove = set()
        for edge in self._game._graph.edges():
            i = 1 if edge[0][0] == 'sys' else 2
            agent_pos = edge[0][i]
            agent_fwd_pos = edge[1][i]
            # remove the direction data fromt the state
            if self.prune_edge_based_on_direction(agent_pos[0], agent_fwd_pos[0], debug=debug, agent_info='sys' if i == 1 else 'env'):
                edges_to_remove.add(edge)
        
        self._game._graph.remove_edges_from(edges_to_remove)

        # might have to call reachability method to remove the unreachable states.
        remove_non_reachable_states(game=self._game, debug=False)
    

    def _extend_trans_init(self):
        # Get the original initial state
        init_state: tuple = self._game.get_initial_states()[0][0]
        # self._game._graph.nodes[init_state]['init'] = False
        assert init_state[0] == 'sys', "[Error] Initial state is not a system state. Please check the initial state."
        new_env_state = deepcopy(init_state)
        tmp = list(new_env_state)
        tmp[0] = 'env' 
        new_env_state = tuple(tmp)
        self._game.add_state(new_env_state, init=False, player='adam', ap='')
        actions = 'stay'
        init_sys_pos = init_state[1][0]

        # Add an edge from this state to the original init state
        self._game.add_edge(init_state, new_env_state,
                            weight=0,
                            actions=actions)
        
        # get the successor state of the init state; ther are multiple then any one is fine.
        succ_state = list(self._game._graph.successors(init_state))[0]
        # get the successor state of the this state
        for edge in self._game._graph.edges(succ_state):
            new_succ = tuple(['sys', ((init_sys_pos), 'right'), ((edge[1][2]), 'right')])
            self._game.add_state(new_succ, init=False, player='eve', ap='')
            _org_edge_attrs = self._game._graph.edges[succ_state, edge[1], 0]
            self._game.add_edge(new_env_state, new_succ, **_org_edge_attrs)
        
        # remove all the other edges from the init state except for the stay one
        remove_edges = set()
        for src_state, dest_state in self._game._graph.edges(init_state):
            if self._game._graph.edges[src_state, dest_state, 0]['actions'] != 'stay':
                remove_edges.add((src_state, dest_state))
        self._game._graph.remove_edges_from(remove_edges)

        
        # # need to add edges from this state
        # for multiactions in self.minigrid_env.player_actions['env']:
            
            
        #     (dest_state, _, _) = self.minigrid_env._make_transition(multiactions, new_env_state)
        #     action_str = self.minigrid_env.ACTION_ENUM_TO_STR[tuple(map(tuple, multiactions))]

        #     possible_edge = (new_env_state, dest_state)
            
        #     # if the agent can not stay in the same cell then do not construct those edges
        #     # if not wait:
        #     i = 0 if new_env_state[0] == 'env' else 1
        #     if new_env_state[1][i] == dest_state[1][i]:
        #         continue


            # (nodes, edges, _, _) = self._add_edge(nodes, edges, 
            #                                      action_str,
            #                                      _, possible_edge)



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
        width=7,
        height=7,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
        env_agent_start_pos=[(1, 2)],
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
        
        # put lava around the goal region
        self.put_obj(Lava(), 2, 2)
        # self.put_obj(Lava(), 3, 2)
        self.put_obj(Lava(), 4, 2)

        # Lava
        self.put_obj(Lava(), 4, 3)
        self.put_obj(Lava(), 4, 4)

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



class NarrowCorridorLavaRAL25(MultiAgentMiniGridEnv):
    """
    Create a minigrid env. 7 by 5 gridworld with Goal on top right. 
     Sys agent start on thr bottom left and Env agent starts on the top right.  A winning stratgey does not exists. 
    """

    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
        env_agent_start_pos=[(6, 1)],
        env_agent_start_dir=[0],
        goal_pos=[(7, 1)],
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
        
        ## Narrow lava corridor
        wall: bool = True
        if wall:
        # horizontal lava
            self.put_obj(Wall(), 1, 4)
            self.put_obj(Wall(), 2, 4)
            self.put_obj(Wall(), 3, 4)

            # vertical lava
            self.put_obj(Wall(), 3, 3)
            self.put_obj(Wall(), 3, 2)

            # horizontal lava
            self.put_obj(Wall(), 2, 2)
            self.put_obj(Wall(), 1, 2)

            # Build the other side
            self.put_obj(Wall(), 7, 2)
            self.put_obj(Wall(), 6, 2)
            self.put_obj(Wall(), 5, 2)

            # vertical lava
            self.put_obj(Wall(), 5, 3)
            self.put_obj(Wall(), 5, 4)

            # horizontal lava
            self.put_obj(Wall(), 6, 4)
            self.put_obj(Wall(), 7, 4)
        else:
            self.put_obj(Water(), 1, 4)
            self.put_obj(Water(), 2, 4)
            self.put_obj(Water(), 3, 4)
            self.put_obj(Water(), 1, 3)
            self.put_obj(Water(), 2, 3)
            

            # vertical lava
            self.put_obj(Water(), 3, 3)
            self.put_obj(Water(), 3, 2)

            # horizontal lava
            self.put_obj(Water(), 2, 2)
            self.put_obj(Water(), 1, 2)

            # Build the other side
            self.put_obj(Water(), 7, 2)
            self.put_obj(Water(), 6, 2)
            self.put_obj(Water(), 5, 2)

            # vertical lava
            self.put_obj(Water(), 5, 3)
            self.put_obj(Water(), 5, 4)

            # horizontal lava
            self.put_obj(Water(), 6, 4)
            self.put_obj(Water(), 7, 4)
            self.put_obj(Water(), 6, 3)
            self.put_obj(Water(), 7, 3)


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
                    restricted_objs=['lava', 'floor', 'water'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal square thorught a narrow lava corridor, don"t touch lava'


class IntruderRobotRAL25(MultiAgentMiniGridEnv):
    """
    Create a minigrid env. 7 by 5 gridworld with Goal on top right. 
     Sys agent start on thr bottom left and Env agent starts on the top right.  A winning stratgey does not exists. 
    """

    def __init__(
        self,
        width=11,
        height=9,
        agent_start_pos=(1, 7),
        agent_start_dir=0,
        env_agent_start_pos=[(8, 1)],
        env_agent_start_dir=[0],
        goal_pos=[(9, 1)],
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
        
        # Human region -Env agent (intruder) can not enter.
        # self.put_obj(Floor(color='purple'), 4, 3)
        # self.put_obj(Floor(color='purple'), 4, 4)
        # self.put_obj(Floor(color='purple'), 4, 5)
        # self.put_obj(Floor(color='purple'), 5, 3)
        self.put_obj(Wall(color='yellow'), 4, 3)
        self.put_obj(Wall(color='yellow'), 4, 4)
        self.put_obj(Wall(color='yellow'), 4, 5)
        self.put_obj(Wall(color='yellow'), 5, 3)
        self.put_obj(Floor(color='purple'), 5, 4) # room
        self.put_obj(Floor(color='purple'), 5, 5) # room door
        # self.put_obj(Floor(color='purple'), 6, 3)
        # self.put_obj(Floor(color='purple'), 6, 4)
        # self.put_obj(Floor(color='purple'), 6, 5)
        self.put_obj(Wall(color='yellow'), 6, 3)
        self.put_obj(Wall(color='yellow'), 6, 4)
        self.put_obj(Wall(color='yellow'), 6, 5)

        # some useful attributes wrt visualization
        # self.agent_view_size
        # gen_obs_grid()

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
                    restricted_objs=['floor'],
                    # restricted_objs=['lava', 'water'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'Robot-intruder game: The task for the robot is to take a photo of the intruder trying \
        to get into the lock and the report it to the human and eventually visit the lock. The intruder can not enter the human"s room.'
    

class TwoNarrowCorridorLavaRAL25(NarrowCorridorLavaRAL25):
    """
      Inherits the base method overide the _gen_grid() method to create two narrow corridors.
    """

    def _gen_grid(self, width, height):
        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place a goal square in the top-left corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)
        
        ## Narrow lava corridor
        # wall: bool = True
        for j in range(1, 3):
            for i in range(2, height - 2):
                self.put_obj(Wall(), j, i)
        
        for j in range(4, 6):
            for i in range(2, height - 2):
                self.put_obj(Wall(), j, i)

        # for j in range(4, 6):
        for i in range(2, height - 2):
            self.put_obj(Wall(), width-2, i)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            # self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
            self.put_agent(ConstrainedAgent(name='SysAgent', restricted_objs=['wall'], view_size=self.view_size), *p, d, True)
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
                    restricted_objs=['lava', 'wall', 'water'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal square thorught a narrow lava corridor, don"t touch lava'


class SmallFourRoomsRobotRAL25(MultiAgentMiniGridEnv):
    """
     4 Four in each corner in a 9 by 9 gridworld. Each room has an opening either to enter the room. 
     Each room has a 1x1 lava bock and the agent can navigate either in clockwise or counter-clockwise direction in each room. 
     The goal is avoid the Env agent while reach the goal region.
    """

    def __init__(
        self,
        width=11,
        height=11,
        agent_start_pos=(5, 1),
        agent_start_dir=0,
        env_agent_start_pos=[(5, 9)],
        env_agent_start_dir=[0],
        goal_pos=[(1, 5)],
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

        # Room are number as per quadrants
        # Room 1 - Lava Block
        self.put_obj(Lava(), 8, 2)  
        

        # Room 1 - Vertical wall
        self.put_obj(Wall(), 4, 1)
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        # self.put_obj(Wall(), 8, 5)
        # Room 1 - Horizontal wall
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 1, 4)

        # Room 2 - Vertical wall
        self.put_obj(Wall(), 6, 1)
        self.put_obj(Wall(), 6, 3)
        self.put_obj(Wall(), 6, 4)
        # Room 2 - Horizontal wall
        self.put_obj(Wall(), 7, 4)
        self.put_obj(Wall(), 9, 4)

        # Room 2 - Lava Block
        self.put_obj(Lava(), 2, 2)        

        # Room 3 - vertical call
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 4, 7)
        self.put_obj(Wall(), 4, 9)
        # self.put_obj(Wall(), 4, 12)
        # self.put_obj(Wall(), 6, 13)
        # Room 3 - Horizontal wall
        self.put_obj(Wall(), 3, 6)
        self.put_obj(Wall(), 1, 6)
        # self.put_obj(Wall(), 2, 8)
        # self.put_obj(Wall(), 1, 8)

        # Room 3 - Lava Block
        self.put_obj(Lava(), 2, 8)
        
        # Room 4 - Vertical wall
        self.put_obj(Wall(), 6, 6)
        self.put_obj(Wall(), 6, 7)
        self.put_obj(Wall(), 6, 9)
        # Room 4 - Horizontal wall
        self.put_obj(Wall(), 7, 6)
        self.put_obj(Wall(), 9, 6)

        # Room 4 - Lava Block
        self.put_obj(Lava(), 8, 8)

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
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    restricted_objs=['floor'],
                    ),
                *p,
                d,
                False)

        self.mission = 'Four Room game with specific direction of travel: The goal is avoid the Env agent while reach the goal region.'



class FourRoomsRobotRAL25(MultiAgentMiniGridEnv):
    """
     4 Four in each corner in 15 by 15 gridworld. Each room has an opening either to enter the room. 
     Each room has a 3x3 lava bock and the agent can navigate either in clockwise or counter-clockwise direction in each room. 
     The goal is avoid the Env agent while reach the goal region.
    """

    def __init__(
        self,
        width=15,
        height=15,
        agent_start_pos=(7, 1),
        agent_start_dir=0,
        env_agent_start_pos=[(7, 13)],
        env_agent_start_dir=[0],
        goal_pos=[(1, 7)],
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

        # Room are number as per quadrants

        # Room 1 - Lava wall
        for i in range(10, 13):
            for j in range(2, 5):
                self.put_obj(Lava(), i, j)  

        # Room 1 - Vertical wall
        self.put_obj(Wall(), 8, 1)
        self.put_obj(Wall(), 8, 2)
        self.put_obj(Wall(), 8, 4)
        self.put_obj(Wall(), 8, 5)
        # Room 1 - Horizontal wall
        self.put_obj(Wall(), 8, 6)
        self.put_obj(Wall(), 9, 6)
        self.put_obj(Wall(), 10, 6)
        self.put_obj(Wall(), 12, 6)
        self.put_obj(Wall(), 13, 6)

        # Room 2 - Vertical wall
        self.put_obj(Wall(), 6, 1)
        self.put_obj(Wall(), 6, 2)
        self.put_obj(Wall(), 6, 4)
        self.put_obj(Wall(), 6, 5)
        # Room 2 - Horizontal wall
        self.put_obj(Wall(), 6, 6)
        self.put_obj(Wall(), 5, 6)
        self.put_obj(Wall(), 4, 6)
        self.put_obj(Wall(), 2, 6)
        self.put_obj(Wall(), 1, 6)

        # Room 2 - Lava wall
        for i in range(2, 5):
            for j in range(2, 5):
                self.put_obj(Lava(), i, j)        

        # Room 3 - vertical call
        self.put_obj(Wall(), 6, 8)
        self.put_obj(Wall(), 6, 9)
        self.put_obj(Wall(), 6, 10)
        self.put_obj(Wall(), 6, 12)
        self.put_obj(Wall(), 6, 13)
        # Room 3 - Horizontal wall
        self.put_obj(Wall(), 5, 8)
        self.put_obj(Wall(), 4, 8)
        self.put_obj(Wall(), 2, 8)
        self.put_obj(Wall(), 1, 8)

        # Room 3 - Lava wall
        for i in range(2, 5):
            for j in range(10, 13):
                self.put_obj(Lava(), i, j)
        
        # Room 4 - Vertical wall
        self.put_obj(Wall(), 8, 8)
        self.put_obj(Wall(), 8, 9)
        self.put_obj(Wall(), 8, 10)
        self.put_obj(Wall(), 8, 12)
        self.put_obj(Wall(), 8, 13)
        # Room 4 - Horizontal wall
        # self.put_obj(Wall(), 8, 7)
        self.put_obj(Wall(), 9, 8)
        self.put_obj(Wall(), 10, 8)
        self.put_obj(Wall(), 12, 8)
        self.put_obj(Wall(), 13, 8)

        # Room 4 - Lava wall
        for i in range(10, 13):
            for j in range(10, 13):
                self.put_obj(Lava(), i, j)

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
                    restricted_objs=['floor'],
                    # restricted_objs=['lava', 'water'],
                    # restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'Four Room game with specific direction of travel: The goal is avoid the Env agent while reach the goal region.'


register(
    id='MiniGrid-LavaComparison_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:CorridorLava'
)

register(
    id='MiniGrid-LavaAdm_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:CorridorLavaAAAI25'
)


register(
    id='MiniGrid-NarrowLavaAdm_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:NarrowCorridorLavaRAL25'
)


register(
    id='MiniGrid-TwoNarrowLavaAdm_karan-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:TwoNarrowCorridorLavaRAL25'
)

register(
    id='MiniGrid-IntruderRobotRAL25-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:IntruderRobotRAL25'
)


register(
    id='MiniGrid-FourRoomsRobotRAL25-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:FourRoomsRobotRAL25'
)


register(
    id='MiniGrid-SmallFourRoomsRobotRAL25-v0',
    entry_point='src.graph_construction.minigrid_two_player_game:SmallFourRoomsRobotRAL25'
)