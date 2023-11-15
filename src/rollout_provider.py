'''
This script contains all Rollout povider classes for different types of Games and agent behaviours (adversarial, cooperative, etc.)
'''
import random
import numpy as np

from typing import List, Union, Dict

from src.graph_construction.two_player_game import TwoPlayerGame
from regret_synthesis_toolbox.src.graph.product import ProductAutomaton

from regret_synthesis_toolbox.src.graph import TwoPlayerGraph
from regret_synthesis_toolbox.src.strategy_synthesis.adversarial_game import ReachabilityGame
from regret_synthesis_toolbox.src.strategy_synthesis.cooperative_game import CooperativeGame
from regret_synthesis_toolbox.src.strategy_synthesis.regret_str_synthesis import RegretMinimizationStrategySynthesis 
from regret_synthesis_toolbox.src.strategy_synthesis.value_iteration import ValueIteration
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_syn import QualitativeBestEffortReachSyn, QuantitativeBestEffortReachSyn
from regret_synthesis_toolbox.src.strategy_synthesis.best_effort_safe_reach import QualitativeSafeReachBestEffort, QuantitativeSafeReachBestEffort


class  RegretStrategyRolloutProvider:
    """
    This class implements the rollout provider for the Franka Robot assigned a task with different types of humans for regret-minimzing strategy.

    In Regret synthesis, Given the product graph, we construct the Graph of utility and then Graph of Best-Response.
    Then we compute the regret minimizing strategy on Graph of Best-Response. Regret Minimizing strategy is memoryless on this graph. Thus, when rolling out, we rollout on this graph.
    """

    def __init__(self, twa_game: ProductAutomaton,  dfa_game: ProductAutomaton, regret_strategy: dict, regret_sv: dict, debug: bool = False) -> None:
        """
        @param dfa_game: is the product graph of the DFA and the game graph. 
        @param regret_strategy: Regret Strategy is the strategy computed by the regret synthesis algorithm. It is a dictionary with keys as states of the DFA Game and values as the Next state.
        @param regret_sv: dfa_game Regret SV is the dictionary with keys as states of the DFA Game and values as the state values.
        """
        self.twa_game: ProductAutomaton = twa_game
        self.dfa_game: ProductAutomaton = dfa_game
        # self.twa_game = regret_strategy.graph_of_alternatives
        self.strategy: Dict[str, str] = regret_strategy
        self.state_values: Dict[str, Union[int, float]] = regret_sv

        self.init_state = self.twa_game.get_initial_states()[0][0]
        self.target_state = self.dfa_game.get_accepting_states()

        self.action_seq: List[str] = []

        self.debug: bool = debug
    

    def _get_successors_based_on_str(self, curr_state) -> str:
        succ_list = []
        succ_state = self.strategy.get(curr_state, None)

        if succ_state is None:
            return
        # __val = __reg_val[__succ_state]
        for count, n in enumerate(list(self.twa_game._graph.successors(curr_state))):
            # if __reg_val[__n] == __val:
            edge_action = self.twa_game._graph[curr_state][n][0].get("actions")
            print(f"[{count}], state:{n}, action: {edge_action}: {self.state_values[n]}")
            succ_list.append(n)

        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]


    def get_manual_rollout(self) -> None:
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

            if next_state[0][0] in self.target_state:
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


    def get_rollout_with_human_intervention(self) -> None:
        """
        This method returns a rollout for the given strategy. The huma

        @param max_coop_actions: Specify the number of cooperative actions the human will take. Default is set to very high number.
        @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks random actions.
        """
        print("Rolling out with human interventions")
        states = []
        states.append(self.init_state)
        curr_state = self.init_state

        for _n in self.twa_game._graph.successors(self.init_state):
            print(f"Reg Val: {_n}: {self.state_values[_n]}")
        next_state = self.strategy[self.init_state]

        self.action_seq.append(self.twa_game._graph[self.init_state][next_state][0].get("actions"))

        while True:
            curr_state = next_state
            states.append(curr_state)
            next_state = self.strategy.get(curr_state, None)

            if next_state[0][0] in self.target_state:
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
        

    def get_rollout_with_epsilon_human_intervention(self, epsilon: float = -1) -> None:
        """
        This method returns a rollout for the given strategy by asuming a cooperative human.

        @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks random actions.

        Epsilon = 0: Env is completely adversarial - Maximizing Sys player's regret
        Epsilon = 1: Env is completely random - Minimizing Sys player's regret
        """
        assert epsilon >= 0 and epsilon <= 1, "Epsilon value should be between 0 and 1"
        if epsilon == 0:
            self.get_rollout_with_human_intervention()
            return None
        
        new_strategy_dictionary = self._compute_epsilon_str_dict(epsilon=epsilon)
        
        self.strategy = new_strategy_dictionary
        self.get_rollout_with_human_intervention()