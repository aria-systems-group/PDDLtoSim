'''
This script contains all Rollout povider classes for different types of Games and agent behaviours (adversarial, cooperative, etc.)
'''
import re

from typing import List, Union
from abc import ABC, abstractmethod

from src.graph_construction.two_player_game import TwoPlayerGame

from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
from regret_synthesis_toolbox.src.graph.product import ProductAutomaton


class RolloutProvider(ABC):
    """
     An abstract class which needs to implemented for various strategy rollouts
    """

    def __init__(self, game: ProductAutomaton, strategy_handle, debug: bool = False, max_steps: int = 10) -> 'RolloutProvider':
        self._game: Union[ProductAutomaton, TwoPlayerGame] = game
        self._game_name: str = game.graph_name
        self._strategy_handle = strategy_handle
        self._strategy: dict = None
        self._env_strategy: dict = None
        self._state_values: dict = None
        self._init_state: list = []
        self._target_states: list = []
        self._sink_states: list = []
        self._absorbing_states: list = []
        self._action_seq: List[str] = []
        self.debug = debug
        self.max_steps: int = max_steps
        self.set_strategy()
        self.set_env_strategy()
        self.set_state_values()
        self.set_target_states()
        self.set_init_states()
        self.set_absorbing_states()
        self.set_sink_states()
        self._check_if_winning()
    

    @property
    def game(self):
        return self._game
    

    @property
    def game_name(self):
        return self._game_name
    
    @property
    def strategy_handle(self):
        return self._strategy_handle
    
    @property
    def strategy(self):
        return self._strategy
    
    @property
    def env_strategy(self):
        return self._env_strategy
    
    @property
    def state_values(self):
        return self._state_values
    
    @property
    def init_state(self):
        return self._init_state
    
    @property
    def target_states(self):
        return self._target_states
    
    @property
    def sink_states(self):
        return self._sink_states
    
    @property
    def absorbing_states(self):
        return self._absorbing_states

    @property
    def action_seq(self):
        return self._action_seq
    
    @game.setter
    def game(self, game: TwoPlayerGraph):
        assert isinstance(game, TwoPlayerGraph), "Please enter a graph which is of type TwoPlayerGraph"
        self._game = game
    
    @game_name.setter
    def game_name(self, name: str):
        self._game_name = name
    
    @abstractmethod
    def set_strategy(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_env_strategy(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_state_values(self):
        raise NotImplementedError
    

    @abstractmethod
    def set_target_states(self):
        raise NotImplementedError

    @abstractmethod
    def set_init_states(self):
        raise NotImplementedError
    

    @abstractmethod
    def set_absorbing_states(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_sink_states(self):
        raise NotImplementedError


    @abstractmethod
    def manual_rollout(self):
        raise NotImplementedError
    

    @abstractmethod
    def rollout_with_human_intervention(self):
        raise NotImplementedError
    
    @abstractmethod
    def rollout_with_strategy_dictionary(self):
        raise NotImplementedError
    

    @abstractmethod
    def rollout_no_human_intervention(self):
        raise NotImplementedError
    

    @abstractmethod
    def rollout_with_epsilon_human_intervention(self):
        raise NotImplementedError


    def _check_human_action(self, edge_action: str) -> bool:
        """
        A function to check if the current action is a human action or not..
        """
        return True if re.search("\\bhuman-move\\b", edge_action) else False