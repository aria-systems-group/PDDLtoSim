import os
import sys
import warnings

from enum import Enum

from gym_minigrid.minigrid import (MiniGridEnv, Grid, Lava, Floor,
                                   Ball, Key, Door, Goal, Wall, Box)

from gym_minigrid.register import register

from icra_examples.safe_adm_game import remove_non_reachable_states
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph, graph_factory, ProductAutomaton

# set path constants
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = ROOT_PATH

# add wombats to my sys path
sys.path.append(f"{ROOT_PATH}/regret_synthesis_toolbox")

from wombats.systems.minigrid import MultiAgentMiniGridEnv, ConstrainedAgent, Agent, MultiObjGrid, Carpet, Water


class ModifiedIntruderRobotGame:

    def __init__(self, game: TwoPlayerGraph, only_augment_obs: bool = False, modify_game: bool = False, debug: bool = False):
         self._game = game
         self._aug_game = None
         self.debug = debug
         if only_augment_obs:
            self._augment_obs()
         if modify_game:
            self.modify_intruder_robot_game()
    
    @property
    def aug_game(self):
        """
        Getter method for the _aug_game attribute.
        :return: The augmented game (product automaton).
        """
        if self._aug_game is None:
            raise ValueError("The augmented game has not been constructed yet. Please call the modify_intruder_robot_game method first.") 
        return self._aug_game
    
    def augement_ap(self, state, ap, game = None):
        if game is None:
            game = self._game
        if game._graph.nodes[state].get('ap') == '':
            game._graph.nodes[state]['ap'] = ap
        elif game._graph.nodes[state]['ap'] not in ["floor_purple_open", "agent_blue_right"] :
            game._graph.nodes[state]['ap'] = game._graph.nodes[state].get('ap') + "__" + ap
    
    def add_door_ap(self):
        """
        A helper function to add the doop label to the AP.
        """
        for state in self._game._graph.nodes():
            # augement ap with label sd/rd
            agent_pos = state[2][0]
            sys_pos = state[1][0]
            if agent_pos == (5, 5):
                aug_ap = "ed"
                self.augement_ap(state, aug_ap, self._game)
            elif sys_pos == (5, 5):
                aug_ap = "rd"
                self.augement_ap(state, aug_ap, self._game)
        
    def _augment_obs(self, game = None):
        """
         A helper method used to augment the AP at each state and see if the Agent observed the Env player or not.
        """
        if game is None:
            game = self._game
        for state in game._graph.nodes():
            # augement ap with agent_observed label
            # the order if elif should chnage as ProductAutomaton is child of TwoPlayerGraph
            if isinstance(game, ProductAutomaton):
                agent_pos = state[0][2][0]
                sys_pos = state[0][1][0]
            
            elif isinstance(game, TwoPlayerGraph):
                agent_pos = state[2][0]
                sys_pos = state[1][0]

            if (0 <= agent_pos[0] - sys_pos[0] <= 2) and (0 <= sys_pos[1] - agent_pos[1] <= 2):
                self.augement_ap(state, "agent_observed", game)
    

    def modify_intruder_robot_game(self):
        """
         A helper function to modify the IntruderRobot game to make it more interesting. In this game the robot/intruder can secure a transition. 
         Who ever secures the transition can traverse through that block.
        """
        self.add_door_ap()
        # take the product of the two transition systems
        door_ts = graph_factory.get('DFA',
                                    graph_name="minigrid_door_trans",
                                    config_yaml=ROOT_PATH + '/regret_synthesis_toolbox/config/test',
                                    save_flag=False,
                                    sc_ltl='',
                                    use_alias=False,
                                    plot=False)

        product_automaton = graph_factory.get("ProductGraph",
                                              graph_name="minigrid_aug_product_graph",
                                              config_yaml=None,
                                              trans_sys=self._game,
                                              observe_next_on_trans=True,
                                              automaton=door_ts,
                                              save_flag=False,
                                              prune=False,
                                              debug=False,
                                              absorbing=False,
                                              finite=False,
                                              plot=False,
                                              pdfa_compose=True)
        
        # now manually remove transition based on which state in the aug product graph the agent is in.
        # if sd is true that the Env agent can not transition to the (5, 5)
        edges_to_be_pruned = set()
        for state in product_automaton._graph.nodes():
            if state[1] == 't1':
                # robot can not transition to the (5, 5)
                if state[0][0] == 'env' and state[0][1][0] == (5, 5):
                    for sys_state in product_automaton._graph.predecessors(state):
                        edges_to_be_pruned.add((sys_state, state))
            
            elif state[1] == 't2':
                # Env can not transition to the (5, 5)
                if state[0][0] == 'sys' and state[0][2][0] == (5, 5):
                    for env_state in product_automaton._graph.predecessors(state):
                        edges_to_be_pruned.add((env_state, state))
        
        if self.debug:
            print(f"# of Edges prunded: {len(edges_to_be_pruned)}")

        product_automaton._graph.remove_edges_from(edges_to_be_pruned)

        print("Removing unreachable states from the augmented game")
        remove_non_reachable_states(game=product_automaton, debug=True)

        # do this aftwards as the ap are sensitive to state labels. 
        # For exmaple ed is a label in Door TS but 'ed' also belong to word oberserved in 'agent_observed' label.
        self._augment_obs(game=product_automaton)

        self._aug_game = product_automaton

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
                corridor_size: int = 1):
        self._game = game
        self._room_2 = self.Room('r2', room_2[0], room_2[1], room_size)
        self._room_1 = self.Room('r1', room_2[0] + room_size + corridor_size + 2, room_2[1], room_size)
        self._room_3 = self.Room('r3', room_2[0], room_2[1] + (room_size + corridor_size + 2), room_size)
        self._room_4 = self.Room('r4', room_2[0] + room_size + corridor_size + 2, room_2[1] + (room_size + corridor_size + 2), room_size)
        self._rooms = [self._room_1, self._room_2, self._room_3, self._room_4]
        self.allowed_moves: dict = {'clockwise' : {'top': (1, 0), 'bottom': (-1, 0), 'left': (0, -1), 'right': (0, 1)},
                                    'counterclockwise' : {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, 1), 'right': (0, -1)}}
        self.room_direction = room_direction
            
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
        self.put_obj(Wall(), 4, 3)
        self.put_obj(Wall(), 4, 4)
        self.put_obj(Wall(), 4, 5)
        self.put_obj(Wall(), 5, 3)
        self.put_obj(Floor(color='purple'), 5, 4) # room
        # self.put_obj(Floor(color='purple'), 5, 5) # room door
        # self.put_obj(Floor(color='purple'), 6, 3)
        # self.put_obj(Floor(color='purple'), 6, 4)
        # self.put_obj(Floor(color='purple'), 6, 5)
        self.put_obj(Wall(), 6, 3)
        self.put_obj(Wall(), 6, 4)
        self.put_obj(Wall(), 6, 5)

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