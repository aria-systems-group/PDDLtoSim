import sys
import math
import random
import warnings

import numpy as np

from typing import List, Tuple, Set, Optional, Dict, Iterable

from utls import deprecated
from src.rollout_str.rollout_provider_if import RolloutProvider

from regret_synthesis_toolbox.src.graph.product import ProductAutomaton
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantiativeRefinedAdmissible
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantitativeGoUAdmissible, QuantitativeGoUAdmissibleWinning

from regret_synthesis_toolbox.src.simulation.simulator import Simulator


class AdmStrategyRolloutProvider(RolloutProvider):
    """
     This class implements rollout provide for Adm strategy synthesis 
    """

    def __init__(self, game: ProductAutomaton, strategy_handle: QuantitativeGoUAdmissible, debug: bool = False,  max_steps: int = 10, logger: Optional[Simulator] = None) -> 'AdmStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps, logger)

    def set_strategy(self):
        pass
    
    def set_env_strategy(self):
        pass

    def set_target_states(self):
        self._target_states: List = self.game.get_accepting_states()

    def set_init_states(self):
        self._init_state = self.game.get_initial_states()[0][0]
    
    def set_absorbing_states(self):
        self._absorbing_states: List = self.game.get_absorbing_state()
        self._absorbing_states = set(self.absorbing_states).union(set(self.target_states))

    def set_sink_states(self):
        self._sink_states: List = self.game.get_trap_states()
    

    def set_state_values(self):
        pass

    def _check_if_winning(self):
        """
         Since Regret Minimizing strategies DO NOT always exists, 
         it is essential to know if the strategy is enforcing task completion, i.e. is it a Winning strategy or not
        """
        if not self.strategy_handle.is_winning(): 
            print("There does not exist a winning strategy!")

    def _get_successors_based_on_str(self, curr_state, avalues: Set[int]) -> str:
        succ_list = []
        is_env_state: bool = True if self.game.get_state_w_attribute(state=curr_state, attribute='player') == 'adam' else False
        idx_count = 0
        for count, n in enumerate(list(self.game._graph.successors(curr_state))):
            edge_action = self.game._graph[curr_state][n][0].get("actions")            
            # check if admissible or not 
            if not is_env_state and self.check_admissibility(source=curr_state, succ=n, avalues=avalues):
                is_sc_strategy: bool = self.check_sc_strategy(curr_state, n, avalues)
                # is_sco_strategy: bool = self.check_sco_strategy(curr_state, n)
                is_wcoop_strategy: bool = self.check_wcoop_strategy(curr_state, n)
                
                # if is_sc_strategy and is_sco_strategy:
                    # print(f"[{idx_count}], Env state:{n}, Sys action: {edge_action}: SCO Adm")
                if is_sc_strategy:
                    print(f"[{idx_count}], Env state:{n}, Sys action: {edge_action}: SC Adm")
                if is_wcoop_strategy:
                    print(f"[{idx_count}], Env state:{n}, Sys action: {edge_action}: WCoop Adm")
                succ_list.append(n)
                idx_count += 1
            elif is_env_state:
                print(f"[{idx_count}], Sys state:{n}, Env action: {edge_action}")
                succ_list.append(n)
                idx_count += 1
        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]
    
    @deprecated
    def check_sco_strategy(self, source: Tuple, succ: Tuple) -> bool:
        """
          A function that check if the an edge is sc admissible or not. 

          TODO: Need to Update the check: 
            If (cVal(h) < aVal(h)) then cVal(h, \sigma) else aVal(h, \sigma) = aVal(h) = cVal(h).
        """
        if self.strategy_handle.coop_winning_state_values[succ] == self.strategy_handle.coop_winning_state_values[source] :
            return True
        
        return False

    
    def check_sc_strategy(self, source: Tuple, succ: Tuple, avalues: Set[int]) -> bool:
        """
          A function that check if the an edge is sc admissible or not
        """
        if self.strategy_handle.coop_winning_state_values[succ] < min(avalues):
            return True
        
        return False

    def check_wcoop_strategy(self, source: Tuple, succ: Tuple) -> bool:
        """
          A function that check if the an edge is wcoop admissible or not
        """
        if self.strategy_handle.winning_state_values[source] == self.strategy_handle.winning_state_values[succ] and \
              self.strategy_handle.coop_winning_state_values[succ] == self.strategy_handle.adv_coop_state_values[source]:
            return True
        
        return False

    def check_admissibility(self, source: Tuple, succ: Tuple, avalues: Set[int]) -> bool:
        """
         A function that check if the an edge is admissible or not when traversing the tree of plays.
        """
        if self.strategy_handle.coop_winning_state_values[succ] < min(avalues):
            return True
        elif self.strategy_handle.winning_state_values[source] == self.strategy_handle.winning_state_values[succ] == \
              self.strategy_handle.coop_winning_state_values[succ] == self.strategy_handle.adv_coop_state_values[source]:
            return True
        
        return False
    

    def manual_rollout(self):
        """
         This method returns a rollout for the given strategy. This method asks human to input the actions during the rollout. Helpful in debugging.
        """
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        avalues = set({self.strategy_handle.winning_state_values[curr_state]})
        next_state = self._get_successors_based_on_str(curr_state, avalues)

        self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))

        steps = 0

        while True:
            curr_state = next_state
            states.append(curr_state)
            avalues.add(self.strategy_handle.winning_state_values[curr_state])
            next_state = self._get_successors_based_on_str(curr_state, avalues=avalues)

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
    
    def _compute_epsilon_str_dict(self, epsilon: float) -> dict:
        """
        A helper method that return the human action as per the Epsilon greedy algorithm.

        Using this policy we either select a random human action with epsilon probability and the human can select the
        optimal action (as given in the str dict if any) with 1-epsilon probability.
        """
        _new_str_dict = self.env_strategy

        for _from_state in _new_str_dict.keys():
            assert self.game.get_state_w_attribute(_from_state, 'player') == "adam", ...
            f"[Error] Encountered {self.game.get_state_w_attribute(_from_state, 'player')} player state in Env player dictionary. Fix This!!!"
            _succ_states: List[tuple] = [_state for _state in self.game._graph.successors(_from_state)]

            # act random
            if np.random.rand() < epsilon:
                _next_state = random.choice(_succ_states)
                _new_str_dict[_from_state] = _next_state

        return _new_str_dict
        
    
    def rollout_with_human_intervention(self):
        pass

    def rollout_with_strategy_dictionary(self):
        pass

    def rollout_no_human_intervention(self):
        pass

    def rollout_with_epsilon_human_intervention(self):
        pass


class AdmWinStrategyRolloutProvider(AdmStrategyRolloutProvider):
    """
     Overrides the base clas''admissibility checking method to compute admissible winning strategies. 
    """
    def __init__(self, game: ProductAutomaton, strategy_handle: QuantitativeGoUAdmissibleWinning, debug: bool = False, max_steps: int = 10, logger: Optional[Simulator] = None) -> 'AdmWinStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps, logger)
    

    def check_sc_strategy(self, source: Tuple, succ: Tuple, avalues: Set[int]) -> bool:
        """
          Override base method for SC checking. There is an addition check for value presevation property.
        """
        if self.strategy_handle.coop_winning_state_values[succ] < min(avalues) and \
              ((source not in self.strategy_handle.winning_region) or (self.strategy_handle.winning_state_values[succ] != math.inf)):
            return True
        
        return False
    
    @deprecated
    def check_sco_strategy(self, source: Tuple, succ: Tuple) -> bool:
        """
         Override base method for SCO checking.  There is an addition check for value presevation property.

         TODO: Need to Update the check: 
            If (cVal(h) < aVal(h)) then cVal(h, \sigma) else aVal(h, \sigma) = aVal(h) = cVal(h) and the additional value-preserving check.
        """
        if self.strategy_handle.coop_winning_state_values[succ] == self.strategy_handle.coop_winning_state_values[source] and \
              ((source not in self.strategy_handle.winning_region) or (self.strategy_handle.winning_state_values[succ] != math.inf)) :
            return True
        
        return False
    

    def check_admissibility(self, source: Tuple, succ: Tuple, avalues: Set[int]) -> bool:
        """
         Update this admissibility check
        """
        if self.strategy_handle.coop_winning_state_values[succ] < min(avalues) and \
            ((source not in self.strategy_handle.winning_region) or (self.strategy_handle.winning_state_values[succ] != math.inf)):
            return True
        elif self.strategy_handle.winning_state_values[source] == self.strategy_handle.winning_state_values[succ] == \
            self.strategy_handle.coop_winning_state_values[succ] == self.strategy_handle.adv_coop_state_values[source]:
            return True
        
        return False


class RefinedAdmStrategyRolloutProvider(AdmStrategyRolloutProvider):
    """
     This class inherits the base class for rolling out Admissible Strategy. In this class we roll out refined version of Adm strategy.

     All admissible str are maximal in dominance order. As such we dont have an informed way to choose amongst such strategie.
       We order the adm strs from highest to lowest as follows: 
     1. If a winning admissible strategy exists then compute Wcoop and choose those strategies.
     2. If a safe-admissible strategy exists, choose that
     3. If a safe-admissible does not exists then play hopeful admissible strategy. 

     A hopeful-admissible str always exists and worst-case scenario is exactly the same as Admissible strategies. 
    """
    def __init__(self, game: ProductAutomaton, strategy_handle: QuantiativeRefinedAdmissible, debug: bool = False, max_steps: int = 10, logger: Optional[Simulator] = None)  -> 'RefinedAdmStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps, logger)
        self.sys_opt_coop_str: Optional[dict] =  self.strategy_handle.coop_optimal_sys_str
        self.env_coop_str: Optional[dict] = self.strategy_handle.env_coop_winning_str
        self.coop_state_values:  Dict[str, float] = self.strategy_handle.coop_winning_state_values
        self.winning_region: set = self.strategy_handle.winning_region
        self.losing_region: set = self.strategy_handle.losing_region
        self.pending_region: set = self.strategy_handle.pending_region
        self._board = ['-','-','-', '-','-','-', '-','-','-']
        
    
    @property
    def sys_opt_coop_str(self):
        return self._sys_opt_coop_str

    @property
    def env_coop_str(self):
        return self._env_coop_str
    
    @property
    def coop_state_values(self):
        return self._coop_state_values

    @property
    def winning_region(self):
        return self._winning_region

    @property
    def losing_region(self):
        return self._losing_region

    @property
    def pending_region(self):
        return self._pending_region
    
    @sys_opt_coop_str.setter
    def sys_opt_coop_str(self, str_dict: dict):
        self._sys_opt_coop_str = str_dict
    
    @env_coop_str.setter
    def env_coop_str(self, str_dict: dict):
        self._env_coop_str = str_dict
    
    @coop_state_values.setter
    def coop_state_values(self, coop_vals: Dict[str, float]):
        self._coop_state_values = coop_vals
    
    @winning_region.setter
    def winning_region(self, winning_states: set):
        self._winning_region = winning_states
    
    @losing_region.setter
    def losing_region(self, losing_states: set):
        self._losing_region = losing_states

    @pending_region.setter
    def pending_region(self, pending_states: set):
        self._pending_region = pending_states

    def set_strategy(self):
        self._strategy = self.strategy_handle.sys_adm_str
    
    def set_env_strategy(self):
        self._env_strategy = self.strategy_handle.env_winning_str

    def set_state_values(self):
        self._state_values =  self.strategy_handle.winning_state_values
    

    def get_edge_action(self, curr_state, succ_state):
        """
         A tiny wrapper around to chekc there exists multiple edged between two states. If yes, then return both those actions
        """
        num_of_edges: int = len(self.game._graph[curr_state][succ_state])
        if num_of_edges == 1:
            return [self.game._graph[curr_state][succ_state][0].get("actions")]
        else:
            return [self.game._graph[curr_state][succ_state][i].get("actions") for i in range(num_of_edges)]
    

    def get_state_type(self, curr_state):
        """
            A helper function that returns the type of state.
        """
        if curr_state in self.winning_region:
            return 'Win'
        elif curr_state in self.pending_region:
            return 'pen'
        elif curr_state in self.losing_region:
            return 'los'
        else:
            warnings.warn(f"[Error] Encourted state {curr_state} that does not belong to any valid region. Fix this!!!")
            sys.exit(-1)
    

    def _log_action(self, curr_state, next_state, counter, str_type):
        """
        Log the action taken during the rollout. There is different from the logger. This is used for printing and debugging purposes..
        """
        _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
        # if self.action_seq[-1] != _edge_act:
        self.action_seq.append(_edge_act)

        if self.debug:
            print(f"Step {counter}: Conf: {curr_state} - {'Robot Act' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'Env Act'}", 
                f"[{str_type if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else ''}] : {self.action_seq[-1]}")
            print("*****************************************************************************************************")
            if 'tic' in self.game_name:
                self.print_board(next_state)
    

    def _finalize_rollout(self):
        """
         Finalize the rollout by printing the action sequence and logging the results.
        """
        if self.debug:
            print("Action Seq:")
            for _action in self.action_seq:
                print(_action)

        print("Done Rolling out")
        

    def log_data(self, counter: int, curr_state, next_state) -> None:
        """
         A helper method that takes as input the current state and next state and extract the relevant information to be logged.
        """
        curr_player: str = 'sys' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'env'
        weight: int = self.game._graph[curr_state][next_state][0]["weight"]
        obs = self.game._graph[curr_state][next_state][0].get('ap', '')
        action = self.game._graph[curr_state][next_state][0].get("actions")
        state_type: str = self.get_state_type(curr_state)

        # log data
        self.logger.log(steps=counter, cost=[weight], obs=obs, action=action, state=curr_state, curr_player=curr_player, state_type=state_type)
    

    def _get_successors_based_on_str(self, curr_state) -> str:
        succ_list = []
        is_winning: bool = True if curr_state in self.winning_region else False
        
        if isinstance(self.strategy_handle, QuantiativeRefinedAdmissible):
            is_hopeful: bool = self.strategy_handle.play_hopeful_game
        
        for count, n in enumerate(list(self.game._graph.successors(curr_state))):
            # edge_action = self.game._graph[curr_state][n][0].get("actions")
            edge_action = self.get_edge_action(curr_state=curr_state, succ_state=n)
            for e in edge_action:
                if is_winning:
                    state_val = self.state_values[n] 
                elif is_hopeful:
                    state_val = self.strategy_handle.hopeful_game.state_value_dict[n]
                else:
                    state_val = self.coop_state_values[n]
                print(f"[{count}], state:{n}: {'Win' if is_winning else 'Pen'}, action: {e}: {state_val}")
            succ_list.append(n)
        
        # preprocess to check in operation
        if not isinstance(self.sys_opt_coop_str[curr_state], list):
            self.sys_opt_coop_str[curr_state] = [self.sys_opt_coop_str[curr_state]]
        
        if self.game.get_state_w_attribute(curr_state, 'player') == "eve":
            for act in self.strategy.get(curr_state):
                if is_winning:
                    print("Sys Strategy: [Wcoop]", act)
                elif act in self.strategy_handle.safe_adm_str.get(curr_state, []):
                    if act in self.sys_opt_coop_str[curr_state]:
                        print("Sys Strategy: [Safe-Adm][Coop Opt]", act)
                    else:
                        print("Sys Strategy: [Safe-Adm]", act)
                else:
                    assert act in self.strategy_handle.hopeful_adm_str[curr_state], f"[Error] {act} is neither Wcoop, safe-adm, or hope-adm. Fix this bug!!!"
                    if act in self.sys_opt_coop_str[curr_state]:
                        print("Sys Strategy: [Hope-Adm][Coop Opt]", act)
                    else:
                        print("Sys Strategy: [Hope-Adm]", act)
        
        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]


    def get_next_state(self, curr_state, rand_adm: bool = False, coop_env: bool = False) -> Tuple[str, str]:
        """
         A helper function that wrap around sys_strategy dictionary. 
         If Sys strategy is deterministic, i.e., there is only one action to take then we return that action. 
         If multiple actions exists, then: 
            If a Wcoop strategy exists then we randomly choose one
            If safe-adm strategy exists, then we randomly choose one
            If Hope-adm strategy exists then we 
        """
        #### Choosing Env action
        if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
            if coop_env:
                if not isinstance(self.env_coop_str.get(curr_state), list):
                    return self.env_coop_str.get(curr_state), ''
                return random.choice(self.env_coop_str.get(curr_state)), ''
            else:
                if not isinstance(self.env_strategy.get(curr_state), list):
                    return self.env_strategy.get(curr_state), ''
                return random.choice(self.env_strategy.get(curr_state)), ''

        #### Choosing Sys action
        self.strategy[curr_state] = list(self.strategy.get(curr_state))
        
        is_winning: bool = True if curr_state in self.winning_region else False
        act = random.choice(self.strategy.get(curr_state))
        if is_winning:
            return act, 'Win' 
        elif act in self.strategy_handle.safe_adm_str.get(curr_state, []):
            return act, 'Safe-Adm'
        else:
            assert act in self.strategy_handle.hopeful_adm_str[curr_state], f"[Error] {act} is neither Wcoop, safe-adm, or hope-adm. Fix this bug!!!"
            return act, 'Hope-Adm'
    

    def manual_rollout(self):
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        next_state = self._get_successors_based_on_str(curr_state)

        self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
        steps: int = 0

        if 'tic' in self.game_name and self.debug:
            self.print_board(next_state)

        while True and steps < self.max_steps:
            curr_state = next_state
            states.append(curr_state)

            next_state = self._get_successors_based_on_str(curr_state)
            if 'tic' in self.game_name and self.debug:
                self.print_board(next_state)
            

            if next_state in self.absorbing_states:
                _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                if self.action_seq[-1] != _edge_act:
                    self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                if 'tic' in self.game_name and self.debug:
                    self.print_board(next_state)
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
    

    def print_board(self, next_state: Tuple):
        # update board
        for idx, cell in enumerate(next_state[0]):
            if cell == 1:
                self._board[idx] = 'R'
            elif cell == 2:
                self._board[idx] = 'H'
        print(self._board[0] + ' | ' + self._board[1] + ' | ' + self._board[2])
        print(self._board[3] + ' | ' + self._board[4] + ' | ' + self._board[5])
        print(self._board[6] + ' | ' + self._board[7] + ' | ' + self._board[8])
    

    def rollout_with_human_intervention(self, coop_env: bool = False):
        """
         This method returns a rollout for the given strategy with human intervention.
        """
        print("Rolling out with human interventions")
        # states = [self.init_state]
        counter: int = 0
        self.logger.reset_episode()

        curr_state = self.init_state
        next_state, str_type = self.get_next_state(curr_state, rand_adm=True, coop_env=coop_env)
        if self.debug:
            print(f"Init State: {curr_state}")

        # self.action_seq.append(self.game._graph[self.init_state][next_state][0].get("actions"))
        # print(f"Step {counter}: Conf: {curr_state} - Robot Act [{str_type}]: {self.action_seq[-1]}")
        # if 'tic' in self.game_name and self.debug:
        #     self.print_board(next_state)
        self._log_action(curr_state, next_state, counter, str_type)
        self.log_data(counter=counter, curr_state=curr_state, next_state=next_state)

        while True and counter < self.max_steps:
            curr_state = next_state
            # states.append(curr_state)
            next_state, str_type = self.get_next_state(curr_state, rand_adm=True, coop_env=coop_env)

            # if next_state in self.target_states or next_state in self.sink_states:
            if curr_state in self.target_states or curr_state in self.sink_states:
                # _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                # # this statement does not hold for minigrid
                # # if self.action_seq[-1] != _edge_act:
                # self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                # counter += 1
                # print(f"Step {counter}: Conf: {curr_state} - {'Robot Act' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'Env Act'}", 
                #     f"[{str_type if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else ''}] : {self.action_seq[-1]}")
                # print("*****************************************************************************************************")
                counter += 1
                self._log_action(curr_state, next_state, counter, str_type)
                self.log_data(counter=counter,curr_state=curr_state, next_state=next_state)
                break
                # if 'tic' in self.game_name and self.debug:
                #     self.print_board(next_state)
                # break

            if next_state is not None:
                # _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                # if self.action_seq[-1] != _edge_act:
                #     self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                counter += 1
                self._log_action(curr_state, next_state, counter, str_type)
                self.log_data(counter=counter, curr_state=curr_state, next_state=next_state)
            
            # if 'tic' in self.game_name and self.debug:
            #     self.print_board(next_state)
            # counter += 1
            # print(f"Step {counter}: Conf: {curr_state} - {'Robot Act' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'Env Act'}", 
            #       f"[{str_type if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else ''}] : {self.action_seq[-1]}")
            # print("*****************************************************************************************************")

            # self.log_data(counter=counter, curr_state=curr_state, next_state=next_state)
        
        self.logger._results.append(self.logger.get_episodic_data())

        # if self.debug:
        #     print("Action Seq:")
        #     for _action in self.action_seq:
        #         print(_action)
        
        # print("Done Rolling out")
        self._finalize_rollout()

    def rollout_with_epsilon_human_intervention(self, epsilon: float) -> None:
        """
        This method returns a rollout for the given strategy by asuming a cooperative human.

        @param epsilon: Set this value to be 0 for purely adversarial behavior or with epsilon probability human picks random actions.

        Epsilon = 0: Env is completely adversarial - Maximizing Sys player's pyaoff
        Epsilon = 1: Env is completely random
        """
        assert epsilon >= 0 and epsilon <= 1, "Epsilon value should be between 0 and 1"
        if epsilon == 0:
            self.rollout_with_human_intervention()
            return None
        
        new_strategy_dictionary = self._compute_epsilon_str_dict(epsilon=epsilon)
        
        self._env_strategy = new_strategy_dictionary
        self.rollout_with_human_intervention()


    def rollout_with_mixed_human_intervention(self, coop_env: bool = False, n_steps: int = 2):
        """
         This method returns a rollout for a human intervening randomly at every n steps.
        """
        print("Rolling out with Mixed human interventions")
        states = [self.init_state]
        counter: int = 0
        curr_state = self.init_state
        next_state, str_type = self.get_next_state(curr_state, rand_adm=True, coop_env=coop_env)
        
        if self.debug:
            print(f"Init State: {curr_state}")

        # self.action_seq.append(self.game._graph[self.init_state][next_state][0].get("actions"))
        # if self.debug:
        #     print(f"Step {counter}: Conf: {curr_state} - Robot Act [{str_type}]: {self.action_seq[-1]}")
        # if 'tic' in self.game_name and self.debug:
        #     self.print_board(next_state)
        self._log_action(curr_state, next_state, counter, str_type)
        env_count = 0
        while True and counter < self.max_steps:
            curr_state = next_state
            states.append(curr_state)
            next_state, str_type = self.get_next_state(curr_state, rand_adm=True, coop_env=coop_env)

            if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
                env_count += 1

            if env_count % n_steps == 0 and self.game.get_state_w_attribute(curr_state, 'player') == "adam":
                next_state = random.choice([_state for _state in self.game._graph.successors(curr_state)])

            if next_state in self.target_states or next_state in self.sink_states:
                # _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
                # # this statement does not hold for minigrid
                # self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
                # counter += 1
                # if self.debug:
                #     print(f"Step {counter}: Conf: {curr_state} - {'Robot Act' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'Env Act'}", 
                #         f"[{str_type if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else ''}] : {self.action_seq[-1]}")
                #     print("*****************************************************************************************************")
                # if 'tic' in self.game_name and self.debug:
                #     self.print_board(next_state)
                counter += 1
                self._log_action(curr_state, next_state, counter, str_type)
                break

            if next_state is not None:
            #     _edge_act = self.game._graph[curr_state][next_state][0].get("actions")
            #     if self.action_seq[-1] != _edge_act:
            #         self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
            
            # if 'tic' in self.game_name and self.debug:
            #     self.print_board(next_state)
                counter += 1
                self._log_action(curr_state, next_state, counter, str_type)
            
            # counter += 1
            # if self.debug:
            #     print(f"Step {counter}: Conf: {curr_state} - {'Robot Act' if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else 'Env Act'}", 
            #         f"[{str_type if self.game.get_state_w_attribute(curr_state, 'player') == 'eve' else ''}] : {self.action_seq[-1]}")
            #     print("*****************************************************************************************************")
   
        
        # if self.debug:
        #     print("Action Seq:")
        #     for _action in self.action_seq:
        #         print(_action)
        
        # print("Done Rolling out")
        self._finalize_rollout()


class RandomSysStrategyRolloutProvider(RefinedAdmStrategyRolloutProvider):
    """
     This class override the Adm Sys strategy and replace it with a random strategy.
    """
    def __init__(self, game, strategy_handle, debug = False, max_steps = 10, logger: Optional[Simulator] = None):
        super().__init__(game, strategy_handle, debug, max_steps, logger)
    
    def get_next_state(self, curr_state, rand_adm: bool = False, coop_env: bool = False) -> Tuple[str, str]:
        """
         A helper function that wraps around sys_strategy and env_strategy dictionary. 
         If Sys strategy is random.
        """
        #### Choosing Env action
        if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
            if coop_env:
                if not isinstance(self.env_coop_str.get(curr_state), list):
                    return self.env_coop_str.get(curr_state), ''
                return random.choice(self.env_coop_str.get(curr_state)), ''
            else:
                if not isinstance(self.env_strategy.get(curr_state), list):
                    return self.env_strategy.get(curr_state), ''
                return random.choice(self.env_strategy.get(curr_state)), ''

        #### Choosing Sys action
        try:
            self.strategy[curr_state] = list(self.strategy.get(curr_state))
            act = random.choice(self.strategy.get(curr_state))
            return act, ''
        except TypeError:
            return random.choice(list(self.game._graph.successors(curr_state))), ''

        


    def set_strategy(self):
        _new_str_dict = self.strategy_handle.sys_adm_str

        for _from_state in self.strategy_handle.sys_adm_str.keys():
            assert self.game.get_state_w_attribute(_from_state, 'player') == "eve", ...
            f"[Error] Encountered {self.game.get_state_w_attribute(_from_state, 'player')} player state in Sys player dictionary. Fix This!!!"
            # act random
            _new_str_dict[_from_state] = [random.choice([_state for _state in self.game._graph.successors(_from_state)])]

        self._strategy = _new_str_dict


class AdmMemeorylessStrRolloutProvider(RefinedAdmStrategyRolloutProvider):

    def __init__(self, game, strategy_handle, debug = False, max_steps = 10, logger: Optional[Simulator] = None):
        super().__init__(game, strategy_handle, debug, max_steps, logger)
    

    def _get_successors_based_on_str(self, curr_state) -> str:
        raise NotImplementedError()
    
    
    def get_next_state(self, curr_state, rand_adm = False, coop_env = False):
        #### Choosing Env action
        if self.game.get_state_w_attribute(curr_state, 'player') == "adam":
            if coop_env:
                if not isinstance(self.env_coop_str.get(curr_state), list):
                    return self.env_coop_str.get(curr_state), ''
                return random.choice(self.env_coop_str.get(curr_state)), ''
            else:
                if not isinstance(self.env_strategy.get(curr_state), list):
                    return self.env_strategy.get(curr_state), ''
                return random.choice(self.env_strategy.get(curr_state)), ''

        #### Choosing Sys action
        self.strategy[curr_state] = list(self.strategy.get(curr_state))
        is_winning: bool = True if curr_state in self.winning_region else False
        act = random.choice(self.strategy.get(curr_state))
        if is_winning:
            return act, 'Win' 
        else:
            return act, 'Coop'
