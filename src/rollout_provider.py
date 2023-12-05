'''
This script contains all Rollout povider classes for different types of Games and agent behaviours (adversarial, cooperative, etc.)
'''
import re
import sys
import random
import warnings
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union

from src.graph_construction.two_player_game import TwoPlayerGame
from regret_synthesis_toolbox.src.graph.product import ProductAutomaton

from utls import timer_decorator
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
# from regret_synthesis_toolbox.src.strategy_synthesis.adversarial_game import ReachabilityGame
# from regret_synthesis_toolbox.src.strategy_synthesis.cooperative_game import CooperativeGame
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import RegretMinimizationStrategySynthesis 
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_safe_reach import QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort


BestEffortClass = Union[QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn, QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort]
Strategy = Union[ValueIteration, RegretMinimizationStrategySynthesis, BestEffortClass]

VALID_ENV_STRINGS = ["manual", "no-human", "random-human", "epsilon-human"]

@timer_decorator
def rollout_strategy(strategy: Strategy,
                     game: ProductAutomaton,
                     debug: bool = False,
                     human_type: str = "random-human",
                     epsilon: float = 0.1,
                     max_iterations: int = 100) -> 'RolloutProvider':
    """
    A function that calls the appropriate rollout provide based on the strategy instance.

    Human_type: input
      "manual" for rollouts with user in the loop
      "no-human" for rollouts with no human intervention
      "random-human" for rollouts with random human intervention
      "epsilon-human" for rollouts with epsilon human intervention
    """
    if isinstance(strategy, RegretMinimizationStrategySynthesis):
        rhandle = RegretStrategyRolloutProvider(game=game,
                                                strategy_handle=strategy,
                                                debug=debug,
                                                max_steps=max_iterations)
    elif isinstance(strategy, ValueIteration):
        rhandle = AdvStrategyRolloutProvider(game=game,
                                             strategy_handle=strategy,
                                             debug=debug,
                                             max_steps=max_iterations)
    
    elif isinstance(strategy, (QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn, QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort)):
        rhandle = BestEffortStrategyRolloutProvider(game=game,
                                                    strategy_handle=strategy,
                                                    debug=debug,
                                                    max_steps=max_iterations)
    else:
        warnings.warn(f"[Error] We do not have rollout provder for strategy of type: {type(strategy)}")
        sys.exit(-1)
    
    if human_type == "manual":
        rhandle.manual_rollout()
    elif human_type == "no-human":
        rhandle.rollout_no_human_intervention()
    elif human_type == "random-human":
        rhandle.rollout_with_human_intervention()
    elif human_type == "epsilon-human":
        rhandle.rollout_with_epsilon_human_intervention(epsilon=epsilon)
    else:
        warnings.warn(f"[Error] Please enter a valid human type from:[ {', '.join(VALID_ENV_STRINGS)} ]")
        sys.exit(-1)

    return rhandle


class RolloutProvider(ABC):
    """
     An abstract class which needs to implemented for various strategy rollouts
    """

    def __init__(self, game: ProductAutomaton, strategy_handle, debug: bool = False, max_steps: int = 10) -> None:
        self._game: Union[ProductAutomaton, TwoPlayerGame] = game
        self._strategy_handle = strategy_handle
        self._strategy: dict = None
        self._state_values: dict = None
        self._init_state: list = []
        self._target_states: list = []
        self._sink_states: list = []
        self._absorbing_states: list = []
        self._action_seq: List[str] = []
        self.debug = debug
        self.max_steps: int = max_steps
        self.set_strategy()
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
    def strategy_handle(self):
        return self._strategy_handle
    
    @property
    def strategy(self):
        return self._strategy
    
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
    
    @abstractmethod
    def set_strategy(self):
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


class RegretStrategyRolloutProvider(RolloutProvider):
    """
    This class implements the rollout provider for the Franka Robot assigned a task with different types of humans for regret-minimzing strategy.

    In Regret synthesis, Given the product graph, we construct the Graph of utility and then Graph of Best-Response.
    Then we compute the regret minimizing strategy on Graph of Best-Response. Regret Minimizing strategy is memoryless on this graph. Thus, when rolling out, we rollout on this graph.
    """
    def __init__(self, game: ProductAutomaton, strategy_handle: RegretMinimizationStrategySynthesis, debug: bool = False,  max_steps: int = 10) -> None:
        super().__init__(game, strategy_handle, debug, max_steps)
        self.twa_game: TwoPlayerGraph = strategy_handle.graph_of_alternatives
    

    def set_strategy(self):
        self._strategy = self.strategy_handle.strategy

    def set_target_states(self):
        self._target_states: List = self.strategy_handle.graph_of_alternatives.get_accepting_states()

    def set_init_states(self):
        self._init_state = self.strategy_handle.graph_of_alternatives.get_initial_states()[0][0]
    
    def set_absorbing_states(self):
        self._absorbing_states: List = self.strategy_handle.graph_of_alternatives.get_absorbing_state()
        self._absorbing_states = set(self.absorbing_states).union(set(self.target_states))

    def set_sink_states(self):
        self._sink_states: List = self.strategy_handle.graph_of_alternatives.get_trap_states()
    
    def set_state_values(self):
        self._state_values = self.strategy_handle.state_values
    
    def _check_if_winning(self):
        """
         Since Regret Minimizing strategies DO NOT always exists, 
         it is essential to know if the strategy is enforcing task completion, i.e. is it a Winning strategy or not
        """
        if not self.strategy_handle.is_winning(): 
            print("[Error] There does not exist a winning strategy!")
            sys.exit(-1)

    def _get_successors_based_on_str(self, curr_state) -> str:
        succ_list = []
        succ_state = self.strategy.get(curr_state, None)

        if succ_state is None:
            return
        
        for count, n in enumerate(list(self.twa_game._graph.successors(curr_state))):
            edge_action = self.twa_game._graph[curr_state][n][0].get("actions")
            print(f"[{count}], state:{n}, action: {edge_action}: {self.state_values[n]}")
            succ_list.append(n)

        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]


    def manual_rollout(self) -> None:
        """
        This method returns a rollout for the given strategy. This method asks human to input the actions during the rollout. Helpful in debugging.
        """
        states = []
        states.append(self.init_state)
        curr_state = self.init_state

        for _n in self.twa_game._graph.successors(self.init_state):
            print(f"Reg Val: {_n}: {self.state_values[_n]}")
        next_state = self._get_successors_based_on_str(curr_state)

        self.action_seq.append(self.twa_game._graph[curr_state][next_state][0].get("actions"))

        while True:
            curr_state = next_state
            states.append(curr_state)
            next_state = self._get_successors_based_on_str(curr_state)

            if next_state in self.target_states:
                _edge_act = self.twa_game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.twa_game._graph[curr_state][next_state][0].get("actions"))
                
                break

            if next_state is not None:
                _edge_act = self.twa_game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.twa_game._graph[curr_state][next_state][0].get("actions"))
                    print(self.action_seq[-1])
        
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)
    

    def rollout_no_human_intervention(self):
        pass


    def rollout_with_human_intervention(self) -> None:
        """
        This method returns a rollout for the given strategy with human intervention.
        """
        print("Rolling out with human interventions")
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        next_state = self.strategy[self.init_state]

        self.action_seq.append(self.twa_game._graph[self.init_state][next_state][0].get("actions"))

        while True:
            curr_state = next_state
            states.append(curr_state)
            next_state = self.strategy.get(curr_state, None)

            if next_state in self.target_states:
                _edge_act = self.twa_game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.twa_game._graph[curr_state][next_state][0].get("actions"))
                break

            if next_state is not None:
                _edge_act = self.twa_game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.twa_game._graph[curr_state][next_state][0].get("actions"))
        
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)
        
        print("Done Rolling out")
    

    def _compute_epsilon_str_dict(self, epsilon: float) -> dict:
        """
        A helper method that return the human action as per the Epsilon greedy algorithm.

        Using this policy we either select a random human action with epsilon probability and the human can select the
        optimal action (as given in the str dict if any) with 1-epsilon probability.
        """
        _new_str_dict = self.strategy

        for _from_state, _ in self.strategy.items():
            if self.twa_game.get_state_w_attribute(_from_state, 'player') == "adam":
                _succ_states: List[tuple] = [_state for _state in self.twa_game._graph.successors(_from_state)]

                # act random
                if np.random.rand() < epsilon:
                    _next_state = random.choice(_succ_states)
                    _new_str_dict[_from_state] = _next_state

        return _new_str_dict
        

    def rollout_with_epsilon_human_intervention(self, epsilon: float) -> None:
        """
        This method returns a rollout for the given strategy by asuming a cooperative human.

        @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks random actions.

        Epsilon = 0: Env is completely adversarial - Maximizing Sys player's regret
        Epsilon = 1: Env is completely random
        """
        assert epsilon >= 0 and epsilon <= 1, "Epsilon value should be between 0 and 1"
        if epsilon == 0:
            self.get_rollout_with_human_intervention()
            return None
        
        new_strategy_dictionary = self._compute_epsilon_str_dict(epsilon=epsilon)
        
        self._strategy = new_strategy_dictionary
        self.rollout_with_human_intervention()


class BestEffortStrategyRolloutProvider(RolloutProvider):
    """
     This class implements rollout provide for Best Effort strategy synthesis 
    """

    def __init__(self, game: ProductAutomaton, strategy_handle: BestEffortClass, debug: bool = False,  max_steps: int = 10) -> None:
        super().__init__(game, strategy_handle, debug, max_steps)

    def set_strategy(self):
        self._strategy = self.strategy_handle.sys_best_effort_str

    def set_target_states(self):
        self._target_states: List = self.game.get_accepting_states()

    def set_init_states(self):
        self._init_state = self.game.get_initial_states()[0][0]
    
    def set_absorbing_states(self):
        self._absorbing_states: List = self.game.get_absorbing_state()
        self._absorbing_states = set(self.absorbing_states).union(set(self.target_states))

    def set_sink_states(self):
        self._sink_states: List = self.game.get_trap_states()
    
    # TODO: Add state value computation to Best-effor synthesis approach
    def set_state_values(self):
        self._state_values = self.strategy_handle.best_effort_state_values


    def _check_if_winning(self):
        """
         Since BEst Effort strategies always exists, it is essential to know if we are playing our Best or enforcing task completion, i.e. Winning stratgey
        """
        if not self.strategy_handle.is_winning():
            print("*************************************************************************************")
            print("[Warning] We are playing Best Effort strategy. There does not exist a winning strategy!")
            print("*************************************************************************************")
        if self.debug:
            print("*************************************************************************************")
            if isinstance(self.strategy_handle, QualitativeSafeReachBestEffort):
                print("We are rolling out Qualitative Safe Reach (Proposed) Best Effort strategy")
            elif isinstance(self.strategy_handle, QuantitativeSafeReachBestEffort):
                print("We are rolling out Quantitative Safe Reach (Proposed) Best Effort strategy")
            elif isinstance(self.strategy_handle, QuantitativeBestEffortReachSyn):
                print("We are rolling out Quantitative Best Effort strategy")
            elif isinstance(self.strategy_handle, QualitativeBestEffortReachSyn):
                print("We are rolling out Org. Qualitative Best Effort strategy")
            print("*************************************************************************************")
    

    def _get_strategy(self, state):
        """
        Tiny wrapper around the strategy dictionary. The strategy Values can be a single state or a set of states. 
        """
        succ_state = self.strategy.get(state, None)

        if isinstance(succ_state, list):
            return random.choice(succ_state)
        elif isinstance(succ_state, set):
            return random.choice(list(succ_state))
        
        return succ_state
    

    def _get_successors_based_on_str(self, curr_state) -> str:
        succ_list = []
        # succ_state = self.strategy.get(curr_state, None)

        # if succ_state is None:
        #     return
        
        for count, n in enumerate(list(self.game._graph.successors(curr_state))):
            edge_action = self.game._graph[curr_state][n][0].get("actions")
            print(f"[{count}], state:{n}, action: {edge_action}: {self.state_values[n]}")
            succ_list.append(n)
        print("Strategy: ", self.strategy.get(curr_state, None))
        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]


    # TODO: Need to add State Value computation and storage capability in Best-Effort synthesis
    def manual_rollout(self):
        # raise NotImplementedError
        states = []
        states.append(self.init_state)
        curr_state = self.init_state

        for _n in self.game._graph.successors(self.init_state):
            print(f"Best Effort Val: {_n}: {self.state_values[_n]}")
        next_state = self._get_successors_based_on_str(curr_state)

        self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))

        steps = 0

        while True and steps < self.max_steps:
            curr_state = next_state
            states.append(curr_state)
            

            # if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
            #     for _succ_state in self.game._graph.successors(curr_state):
            #         _edge_action = self.game._graph[curr_state][_succ_state][0]["actions"]
            #         if not self._check_human_action(_edge_action):
            #             next_state = _succ_state
            #             break
            
            # else:
            next_state = self._get_successors_based_on_str(curr_state)

            if next_state in self.absorbing_states:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                
                break

            if next_state is not None:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                    print(self.action_seq[-1])
            
            steps += 1
        
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)
    
    def rollout_no_human_intervention(self):
        print("Rolling out with No env interventions")
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        next_state = self._get_strategy(self.init_state)

        self.action_seq.append(self.game._graph[self.init_state][next_state][0].get("actions"))

        steps = 0

        while True and steps < self.max_steps:
            curr_state = next_state
            states.append(curr_state)
            next_state = self._get_strategy(curr_state)

            # this is for non-minigrid envs
            if self.game.get_state_w_attribute(curr_state, 'player') == "adam" and 'minigrid' not in self.game._graph.name:
                for _succ_state in self.game._graph.successors(curr_state):
                    _edge_action = self.game._graph[curr_state][_succ_state][0]["actions"]
                    if not self._check_human_action(_edge_action):
                        next_state = _succ_state
                        break
            
            # # for minigrid envs
            # if next_state is None and self.game.get_state_w_attribute(curr_state, 'player') == "adam" and 'minigrid' in self.game._graph.name:
            #     for _succ_state in self.game._graph.successors(curr_state):
            #         _edge_action = self.game._graph[curr_state][_succ_state][0]["actions"]
            #         if not self._check_human_action(_edge_action):
            #             next_state = _succ_state
            #             break


            if next_state in self.absorbing_states:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                break

            if next_state is not None:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
            
            steps += 1
        
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)
        
        print("Done Rolling out")
        

    def rollout_with_human_intervention(self):
        print("Rolling out with human interventions")
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        next_state = self._get_strategy(self.init_state)

        self.action_seq.append(self.game._graph[self.init_state][next_state][0].get("actions"))

        steps = 0

        while True and steps < self.max_steps:
            curr_state = next_state
            states.append(curr_state)
            next_state = self._get_strategy(curr_state)

            if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
                _succ_states: List[tuple] = [_state for _state in self.game._graph.successors(curr_state)]
                next_state = random.choice(_succ_states)

            if next_state in self.absorbing_states:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                # break

            if next_state is not None:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
            
            steps += 1
        
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)
        
        print("Done Rolling out")
    

    def _compute_epsilon_str_dict(self, epsilon: float) -> dict:
        """
        A helper method that return the human action as per the Epsilon greedy algorithm.

        Using this policy we either select a random human action with epsilon probability and the human can select the
        optimal action (as given in the str dict if any) with 1-epsilon probability.
        """
        _new_str_dict = self.strategy

        for _from_state, _ in self.strategy.items():
            if self.game.get_state_w_attribute(_from_state, 'player') == "adam":
                _succ_states: List[tuple] = [_state for _state in self.game._graph.successors(_from_state)]

                # act random
                if np.random.rand() < epsilon:
                    _next_state = random.choice(_succ_states)
                    _new_str_dict[_from_state] = _next_state

        return _new_str_dict
    

    def rollout_with_epsilon_human_intervention(self, epsilon: float) -> None:
        """
        This method returns a rollout for the given strategy by asuming a cooperative human.

        @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks random actions.

        Epsilon = 0: Env is completely adversarial - Maximizing Sys player's regret
        Epsilon = 1: Env is completely random
        """
        assert epsilon >= 0 and epsilon <= 1, "Epsilon value should be between 0 and 1"
        if epsilon == 0:
            self.rollout_with_human_intervention()
            return None
        
        new_strategy_dictionary = self._compute_epsilon_str_dict(epsilon=epsilon)
        
        self._strategy = new_strategy_dictionary
        self.rollout_with_human_intervention()


class AdvStrategyRolloutProvider(BestEffortStrategyRolloutProvider):
    """
     This class implements inherits the Best Effort rollout provider
    """
    def __init__(self, game: ProductAutomaton, strategy_handle, debug: bool = False, max_steps: int = 10) -> None:
        super().__init__(game, strategy_handle, debug, max_steps)
    

    def set_strategy(self):
        self._strategy = self.strategy_handle.str_dict
    
    def set_state_values(self):
        self._state_values =  self.strategy_handle.state_value_dict

    def _check_if_winning(self):
        """
         Since Min-Max strategies DO NOT always exists, it is essential to know
           if we are playing a strategy that is enforcing task completion, i.e. Winning strategy or not.
        """
        if not self.strategy_handle.is_winning(): 
            print("[Error] There does not exist a winning strategy!")
            sys.exit(-1)
