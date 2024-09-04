import math

from typing import List, Tuple, Set, Optional, Dict

from utls import deprecated
from src.rollout_str.rollout_provider_if import RolloutProvider

from regret_synthesis_toolbox.src.graph.product import ProductAutomaton
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantiativeRefinedAdmissible
from regret_synthesis_toolbox.src.strategy_synthesis.adm_str_syn import QuantitativeGoUAdmissible, QuantitativeGoUAdmissibleWinning


class AdmStrategyRolloutProvider(RolloutProvider):
    """
     This class implements rollout provide for Adm strategy synthesis 
    """

    def __init__(self, game: ProductAutomaton, strategy_handle: QuantitativeGoUAdmissible, debug: bool = False,  max_steps: int = 10) -> 'AdmStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps)

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
    def __init__(self, game: ProductAutomaton, strategy_handle: QuantitativeGoUAdmissibleWinning, debug: bool = False, max_steps: int = 10) -> 'AdmWinStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps)
    

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

     All admissible str are maximal in dominance order. As such we an informed way to choose amongst such strategie.
       We order the an adm str from highest to lowest as follows: 
     1. If a winning admissible strategy exists then compute Wcoop and choose those strategies.
     2. If a safe-admissible strategy exists, choose that
     3. If a safe-admissible does not exists then play hopeful admissible strategy. 

     A hopeful-admissible str always exists and worst-case scenario is exactly the same as Admissible strategies. 
    """

    def __init__(self, game: ProductAutomaton, strategy_handle: QuantiativeRefinedAdmissible, debug: bool = False, max_steps: int = 10) -> 'RefinedAdmStrategyRolloutProvider':
        super().__init__(game, strategy_handle, debug, max_steps)
        self._env_coop_str: Optional[dict] = None
        self._coop_state_values:  Dict[str, float] = strategy_handle.coop_winning_state_values
        self._winning_region: set = self.strategy_handle.winning_region
        self._losing_region: set = self.strategy_handle.losing_region
        self._pending_region: set = self.strategy_handle.pending_region
        

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
    
    @env_coop_str.setter
    def env_coop_str(self):
        self._env_coop_str = self.strategy_handle.env_winning_str
    
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
    

    def _get_successors_based_on_str(self, curr_state) -> str:
        succ_list = []
        # succ_state = self.strategy.get(curr_state, None)

        # if succ_state is None:
        #     return
        is_winning: bool = True if curr_state in self.winning_region else False
        for count, n in enumerate(list(self.game._graph.successors(curr_state))):
            edge_action = self.game._graph[curr_state][n][0].get("actions")
            print(f"[{count}], state:{n}: {'Win' if is_winning else 'Pen'}, action: {edge_action}: {self.state_values[n] if is_winning else self.coop_state_values[n]}")
            succ_list.append(n)
        
        if self.game.get_state_w_attribute(curr_state, 'player') == "eve":
            for act in self.strategy.get(curr_state):
                if is_winning:
                    print("Sys Strategy: [Win]", act)
                elif act in self.strategy_handle.safe_adm_str.get(curr_state, []):
                    print("Sys Strategy: [Safe-Adm]", act)
                else:
                    assert act in self.strategy_handle.hopeful_adm_str[curr_state], f"[Error] {act} is neither Wcoop, safe-adm, or hope-adm. Fix this bug!!!"
                    print("Sys Strategy: [Hope-Adm]", act)
        
        idx_num = input("Enter state to select from: ")
        print(f"Choosing state: {succ_list[int(idx_num)]}")
        return succ_list[int(idx_num)]
    

    def manual_rollout(self):
        states = []
        states.append(self.init_state)
        curr_state = self.init_state
        next_state = self._get_successors_based_on_str(curr_state)

        self.action_seq.append(self.game._graph[curr_state][next_state][0].get("actions"))
        steps: int = 0

        while True and steps < self.max_steps:
            curr_state = next_state
            states.append(curr_state)

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